# Tensor API

## Purpose and mental model

This reference documents the public model contracts that are implemented today. The mutable
`Tensor` now connects stable logical metadata to an optional borrowed host-storage association,
and `TensorFactory` provides the public construction boundary for completed descriptors, copied
primitive-array import, independent dense constants including full-value and rectangular identity
tensors, and integer ranges. `TensorRandoms` is the sole public owner of eager normal,
continuous-uniform, bounded-integral, and Bernoulli initialization from an explicit caller-owned
source. `GraphRngState` separately represents an explicit, opaque random-number-generator (RNG)
state occurrence for later graph execution; it neither exposes a numerical Tensor nor performs a
random draw. `Tensor.dropout(probability, state)` constructs one training-only inverted-dropout
occurrence and returns its public output together with the explicitly threaded next state. The
current concrete expression surface contains seven tensor-to-tensor binary arithmetic
methods, of which ADD, SUB, MUL, MIN, and MAX also accept signed integral operands; six floating
or signed-integral tensor-to-tensor comparison methods; three BOOL-only logical
methods, nineteen floating unary elementwise methods, three floating-classification methods, seven
exact-typed plus seven exact-FLOAT64 scalar arithmetic methods, and six range/one-bound clamp
methods, plus one static conditional-selection method and one explicit cast method. Fifteen
numeric aggregate methods add full, one-axis, and retained-axis `sum`, `mean`, `prod`, reduction
`min`, and reduction `max` expression construction: mean remains floating-only, while the other
four families also accept signed-integral input. Six BOOL methods provide full/single-axis
`all`/`any`; ten additional numeric and four BOOL methods add ordered distinct multi-axis forms.
Twelve floating methods add first-class log-sum-exp, corrected
variance/standard deviation, and L1/L2 norms. Six axis-only `argMin`/`argMax` methods add fixed
INT64 index results. Two one-axis `cumSum` methods add shape-preserving numeric
scan expressions, and one-axis `softmax` and `logSoftmax` add shape-preserving floating
normalization expressions. The parameterless `contiguous` method adds a shape-preserving request
for canonical dense row-major result geometry. Two `reshape` overloads add ordered-element-
preserving coordinate changes from either raw `long...` dimensions or an exact normalized
`Shape`. Two `expand` overloads add directional right-aligned singleton and leading-axis
repetition with locally derived zero-stride view geometry when possible. `permute(int...)` adds
arbitrary complete axis reordering, and `transpose()` adds its rank-two `[1, 0]` convenience.
`expandDims(int)` inserts one singleton axis, while `squeeze(int)` removes one selected statically
known singleton axis. `slice(long[], long[], int[], long[])` adds general signed-step directional
half-open selection. `sliceAxis(int, long, long)` supplies its one-axis step-one convenience,
`sliceAxis(int, long, long, long)` supplies an explicit signed step, and `flip(int...)` reverses
explicit axes through one slice occurrence.
`select(int, long)` fixes one scalar coordinate and removes its source axis, with locally derived
logical view geometry when the input layout is resolved and the result is non-empty.
`gather(Tensor, int)` replaces one data axis with the complete indices Shape.
`gatherElements(Tensor, int)` retains the exact same-rank indices Shape after checking every
non-selected Dimension. These are the complete public axis-index gather spellings.
`oneHot(long)` treats an INT32 or INT64 receiver as logical indices and appends one positive
static class axis. It produces storage-free BOOL metadata for one first-class `ONE_HOT`
occurrence without reading any index value.
`gatherNd(Tensor)` and `gatherNd(Tensor, int)` use coordinate tuples to index several data axes
after zero or more structurally equal shared batch Dimensions. They require a static positive
tuple depth, derive the result from the indices prefix plus the untouched data suffix, and never
inspect index values.
The two `scatterElements` overloads construct functional same-rank axis-scatter expressions from
ordered `[data, indices, updates]` inputs. They preserve the exact data Shape and type, combine
data/update gradient eligibility, and never inspect values or mutate the data input.
The three `scatterNd` overloads construct functional tuple-index scatter expressions with
replacement or an explicit reduction and with zero or an explicit shared-batch count. They apply
the Gather-ND result-Shape formula to `updates`, preserve exact data metadata in a fresh unresolved
result, and never inspect values or mutate an input.
`pad(long[], long[], ScalarValue)` adds exactly typed constant-filled positions around every axis;
its retained `double` overload is an exact-FLOAT64 convenience. `tile(long...)` repeats each
complete input pattern along every axis. Static `concat` joins an
ordered non-empty input sequence along an existing axis, static `stack` inserts a count axis for
same-shaped inputs, and instance `unstack` returns an immutable ordered list of independent scalar
select expressions. `unfold` adds general-axis window materialization, while `unfold2d` and
`fold2d` add NCHW im2col and overlap-summing col2im Shape construction. General-axis `FOLD_AXIS`
remains representable as compiler-only model semantics but has no public Tensor method.
Typed access, other expression families, gradient objects and publication behavior,
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

`SliceKind.SLICE` is also a current one-input semantic identity. It pairs with `SliceAttrs`, whose
four immutable parallel lists store normalized starts, selected lengths, distinct axes, and signed
non-zero steps. Entry `i` names the finite sequence `start + k * step` for
`0 <= k < length`, without an exclusive-end sentinel. Public general and single-axis slicing
normalize raw directional half-open bounds against selected static dimensions; `flip(int...)`
constructs the same negative-step semantics directly. Positive non-empty slices may derive logical
view geometry, while every negative-step result remains layout-unresolved under the current
non-negative-stride descriptor. Gradients, compiler behavior, materialization, backend lowering,
ONNX mapping, and execution remain planned in their owning layers.

`PadKind.PAD` with `PadAttrs` and `TileKind.TILE` with `TileAttrs` are current semantic contracts.
Padding attributes store immutable ordered before/after widths plus one exact typed scalar value.
Tiling attributes store immutable positive complete-pattern repeat counts. Empty lists
describe rank-zero scalar identity parameters. Public `Tensor.pad` and `Tensor.tile` now own exact
rank validation, checked result-Shape derivation, unresolved result layout, exact metadata
retention, and fresh one-input provenance. The semantic values themselves still do not construct
Tensors or define conversion of a padding constant, gradients, materialization, compiler or
backend behavior, ONNX mapping, or execution.

`TensorCompositionKind.CONCAT` and `STACK` are current semantic identities.
`CompositionAxisAttrs` stores one normalized non-negative axis shared by concat and stack, while
the kind distinguishes an existing concat axis from a newly inserted stack result-axis position.
Public composition methods, input and result validation, Shape derivation, immutable result
collection construction, and provenance attachment are current. Public unstack is an ordered
convenience over scalar `SELECT`, so every result has its own independent single-output producer
and provenance output index zero; compiler capture or decomposition, gradients,
materialization, and execution remain planned.

`SelectKind.SELECT` is a current scalar-index semantic identity. It pairs with
`SelectAttrs(axis, index)`, whose normalized non-negative axis and scalar coordinate fix one
logical position on one source axis and remove that axis from the conceptual result. For source
Shape `[2, 3, 4]`, axis `1` and index `2` therefore mean conceptual result Shape `[2, 4]`.
The attributes contain no input Shape, so they cannot prove that the axis exists or that the index
is in bounds. Public `Tensor.select` now owns caller-facing negative normalization, local bounds
validation, result Shape/layout construction, and exact one-input provenance. Gradients, compiler
capture, materialization, backend lowering, and execution remain planned.

`AxisGatherKind.GATHER` and `GATHER_ELEMENTS` are current tensor-index semantic identities. Each
pairs with `IndexAxisAttrs`, which stores one already normalized non-negative data-axis position,
and each has ordered logical inputs `[data, indices]`. Their index alignment and conceptual result
Shapes deliberately differ. These semantic values do not inspect either
input, validate index data type or Shape compatibility, construct a Tensor or provenance, define
gradients, or execute indexing. Public `Tensor.gather` and `gatherElements` own raw-axis
normalization, exact `INT32`/`INT64` index representation, family-specific structural Shape checks
and construction, and exact ordered two-input provenance. Both consume an existing index Tensor.
None of the methods interprets index values or checks their bounds.

`OneHotKind.ONE_HOT` is the current index-encoding semantic identity. It pairs with
`OneHotAttrs(depth)`, which stores one positive static trailing-axis extent, and consumes ordered
logical input `[indices]`. Public `Tensor.oneHot(depth)` requires exact INT32 or INT64 index
metadata, preserves every receiver Dimension reference, appends one fresh
`StaticDimension(depth)`, and constructs a fixed-BOOL, non-differentiable result. Construction
reads no values. Valid eventual execution requires every index `i` to satisfy `0 <= i < depth`;
invalid values do not wrap, clamp, select a default, or produce an all-false row. Compiler
capture, constant analysis, bounds enforcement, lowering, backend support, and execution are not
implemented by this model contract.

`GatherNdKind.GATHER_ND` is the current tuple-index semantic identity. It pairs with
`GatherNdAttrs(batchDimensions)`, which stores only the normalized non-negative count of shared
leading batch Dimensions. Ordered logical inputs are `[data, indices]`; the final indices
Dimension supplies tuple depth `K`, so that occurrence-specific fact is not duplicated in the
attributes. Public `Tensor.gatherNd` construction now owns exact `INT32`/`INT64` index-type checks,
input-rank and batch compatibility, structural batch-prefix equality, static positive tuple-depth
validation, result Shape construction, and exact ordered provenance. Index-value bounds,
gradients, compiler behavior, backend behavior, and execution remain planned or separately owned.

`AxisScatterKind.SCATTER_ELEMENTS` is the current functional axis-scatter semantic identity. It
has ordered logical inputs `[data, indices, updates]` and conceptually produces a new result with
the exact `data` Shape without mutating `data` in place. It uses
`ScatterElementsAttrs(axis, reduction)` and the reusable
`ScatterReduction` vocabulary `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`. Replacement with `NONE`
requires unique target coordinates; value-aware duplicate detection is deferred. Public Tensor
construction now owns caller-axis normalization, exact index and data/update type checks, its
same-rank Shape rule, result metadata, and ordered provenance. Index-value bounds,
duplicate detection, gradients, compiler behavior, backend behavior, and execution remain planned
or separately owned.

`ScatterNdKind.SCATTER_ND` is the current functional tuple-index scatter semantic identity. It
pairs with `ScatterNdAttrs(batchDimensions, reduction)`, which stores an already normalized
non-negative shared-batch count and one explicit `ScatterReduction`. Ordered logical inputs are
`[data, indices, updates]`; the final indices Dimension supplies tuple depth `K`, and updates have
the Shape that Gather-ND would read from the same data and indices. The conceptual result is a new
value with the exact data Shape, so the operation does not mutate `data`. Public Scatter-ND Tensor
construction now owns exact index and update type checks, reduction eligibility, input-rank and
batch compatibility, structural batch-prefix equality, static positive tuple depth, exact updates-
Shape validation, result metadata, and ordered provenance. Index bounds, `NONE` duplicate
detection, gradients, compiler behavior, backend behavior, and execution remain planned or
separately owned.

`WindowTransformKind.UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, and `FOLD2D` are current semantic
identities with normalized immutable window attributes. Public `Tensor.unfold`, `unfold2d`, and
`fold2d` own their existing static Shape and compatibility checks, checked window arithmetic,
unresolved result layout, and exact one-input provenance. `FOLD_AXIS` and `FoldAxisAttrs` remain
compiler-only semantic values with no public Tensor construction path; task 0023 owns their first
compiler-generated use. These contracts describe materialized windows or overlap-summing
scatter-add without reading values, attaching storage, capturing a graph, generating gradients,
or executing.

The current types describe logical values, construct storage-free arithmetic, comparison, boolean
logical, conditional-selection, unary, scalar, and value- or index-producing aggregate
expressions, and provide one bounded host-allocation path without executing a tensor:

```text
DataType + Shape + optional LayoutDescriptor + requiresGrad = TensorDescriptor
DataType + physical capacity + exact MemorySegment           = MemorySegmentStorage
Operation + ordered inputs + ordered output descriptors      = TensorProducer
TensorProducer + zero-based output index                     = TensorProvenance
TensorId + exact output descriptor + label + optional provenance + optional host storage = Tensor
TensorFactory + completed descriptor + optional metadata     = public Tensor construction
TensorFactory + resolved descriptor                          = exact-span heap storage + Tensor
TensorFactory + dense descriptor + matching flat array       = copied populated heap Tensor
TensorFactory + rectangular nested primitive array           = inferred dense descriptor + copied Tensor
TensorFactory + scalar/shape/type or Tensor template          = independent dense constant Tensor
TensorFactory + static shape + exact primitive value          = independent dense full-value Tensor
TensorFactory + rows + columns + DataType                     = independent dense identity matrix
TensorFactory + integral bounds/step                          = copied dense INT32 or INT64 range Tensor
TensorRandoms + static shape/type + caller RandomGenerator    = copied dense normal/uniform Tensor
TensorRandoms + static shape + primitive bounds + source      = copied dense INT32/INT64 random Tensor
TensorRandoms + static shape + probability + source           = copied dense BOOL random Tensor
GraphRngState.initial(key, counter)                            = opaque storage-free state expression
left Tensor + binary kind + right Tensor                       = fresh descriptor + provenance Tensor
left Tensor + comparison kind + right Tensor                   = fresh BOOL descriptor + provenance Tensor
left BOOL Tensor + AND/OR + right BOOL Tensor                   = fresh broadcast BOOL expression Tensor
BOOL Tensor + NOT                                               = fresh shape-preserving BOOL expression Tensor
BOOL condition + true/false floating branches                  = fresh broadcast selection Tensor
input Tensor + unary kind                                      = fresh descriptor + provenance Tensor
floating Tensor + classification kind                         = fresh BOOL descriptor + provenance Tensor
input Tensor + scalar kind + matching ScalarValue attributes  = fresh descriptor + provenance Tensor
input Tensor + target DataType                                 = fresh explicit cast Tensor
numeric Tensor + SUM/PROD/MIN/MAX + full/axis attributes         = fresh reduced-shape Tensor
floating Tensor + MEAN + full/axis attributes                   = fresh reduced-shape Tensor
BOOL Tensor + ALL/ANY + full/axis attributes                    = fresh reduced-shape BOOL Tensor
numeric Tensor + ARG_MIN/ARG_MAX + axis/tie attributes          = fresh reduced-shape INT64 Tensor
numeric Tensor + CUM_SUM + axis/mode attributes                 = fresh shape-preserving Tensor
floating Tensor + SOFTMAX/LOG_SOFTMAX + axis attributes        = fresh shape-preserving Tensor
Tensor + CONTIGUOUS                                = fresh static-resolved or dynamic-unresolved Tensor
Tensor + raw/exact target Shape + RESHAPE          = fresh conditional-view reshape Tensor
Tensor + raw/exact target Shape + EXPAND           = fresh conditional-view expand Tensor
Tensor + complete output-to-input axes + PERMUTE   = fresh conditional-view permute Tensor
Tensor + insertion axis + EXPAND_DIMS              = fresh conditional-view rank-expanded Tensor
Tensor + static singleton axis + SQUEEZE           = fresh conditional-view rank-reduced Tensor
Tensor + parallel bounds/axes/signed steps + SLICE = fresh conditional-layout sliced Tensor
Tensor + one axis/bounds/optional step + SLICE     = fresh single-axis sliced Tensor
Tensor + explicit axes + SLICE                     = fresh one-occurrence flipped Tensor
Tensor + one axis and scalar coordinate + SELECT  = fresh conditional-view rank-reduced Tensor
Tensor + arbitrary indices + GATHER               = fresh inserted-Shape gather Tensor
Tensor + same-rank aligned indices + GATHER_ELEMENTS = fresh exact-indices-Shape Tensor
INT32/INT64 indices Tensor + positive depth + ONE_HOT = fresh trailing-axis BOOL Tensor
Tensor + coordinate-tuple indices + GATHER_ND    = fresh indices-prefix-plus-data-suffix Tensor
Tensor + same-rank indices/updates + SCATTER_ELEMENTS = fresh data-Shape replacement/reduction Tensor
Tensor + coordinate-tuple indices/updates + SCATTER_ND = fresh data-Shape replacement/reduction Tensor
Tensor + axis/size/step + UNFOLD_AXIS              = fresh general-axis window Tensor
rank-four NCHW Tensor + window geometry + UNFOLD2D = fresh canonical column Tensor
rank-three columns + NCHW Shape/window + FOLD2D    = fresh NCHW fold Tensor
TensorId / NodeId / ValueId                                  = distinct identity domains
Operation                                                     = OperationKind + OperationAttrs
BinaryArithmeticKind                                          = seven parameterless binary arithmetic semantics
BinaryComparisonKind                                          = six parameterless ordered comparison semantics
BooleanLogicalKind                                            = three parameterless boolean logical semantics
WhereSelectionKind                                            = one parameterless ternary conditional-selection semantic
CastKind + CastAttrs                                          = explicit cast identity + target DataType
AggregateReductionKind + reduction attributes                = full/single-/multi-axis aggregate + statistics + arg extrema
SUM/MEAN + MaskedReductionAttrs                              = masked axis semantics + ordinary broadcast mask
CumulativeSumKind + CumulativeSumAttrs                       = shape-preserving cumulative-sum scan semantics
SoftmaxKind + SoftmaxAttrs                                   = shape-preserving probability normalization semantics
ContiguousKind                                               = parameterless canonical dense row-major geometry request
ShapeTransformKind + TargetShapeAttrs                        = reshape/expand meaning + normalized target Shape
AxisTransformKind + PermutationAttrs / AxisTransformAttrs    = axis reorder/insertion/removal semantics
SliceKind + SliceAttrs                                      = finite signed-step coordinate-sequence semantics
PadKind + PadAttrs                                          = constant-padding meaning + normalized widths/typed constant
TileKind + TileAttrs                                        = complete-pattern per-axis tiling + positive repeat counts
TensorCompositionKind + CompositionAxisAttrs                = concat/stack meaning + normalized axis
SelectKind + SelectAttrs                                   = scalar coordinate + normalized source axis/index
AxisGatherKind + IndexAxisAttrs                            = gather/gather-elements meanings + normalized data axis
OneHotKind + OneHotAttrs                                 = one-hot meaning + positive static depth
GatherNdKind + GatherNdAttrs                              = tuple-index meaning + normalized shared batch count
AxisScatterKind + ScatterElementsAttrs                    = functional scatter-elements meaning + axis/reduction
ScatterNdKind + ScatterNdAttrs                             = functional tuple-scatter meaning + batch count/reduction
WindowTransformKind + window attributes                      = unfold/fold meaning + normalized geometry
UnaryElementwiseKind                                          = nineteen parameterless unary elementwise semantics
FloatingClassificationKind                                   = three parameterless floating classifications
ScalarElementwiseKind                                         = eight parameterized one-input scalar semantics
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
Its deterministic range methods create non-empty exclusive-end `INT32` and `INT64` tensors,
synthesize canonical dense descriptors, build one complete temporary carrier, and delegate final
storage population and identity assignment to flat import. Prefix preparation is test-fixture
mechanics, not a production TensorFactory capability.
Public `TensorRandoms` owns normal, continuous-uniform, bounded-integral, and Bernoulli eager leaf
creation. Each method builds one exact carrier but consumes a transient caller-owned
`RandomGenerator` instead of retaining, selecting, or seeding a source. Primitive `int` or `long`
bounds infer `INT32` or `INT64`, respectively; Bernoulli output is always BOOL; and all integral
and boolean results disable gradients. This stateless namespace is not a random service,
lifecycle owner, source abstraction, seed API, distribution hierarchy, or graph RNG contract.

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
boolean all and any reductions, floating log-sum-exp, variance, standard deviation, L1/L2 norm,
and index-producing arg-min and arg-max. `AxisReductionAttrs` carries one
already normalized non-negative axis plus a retained-dimension choice for ordinary single-axis
forms. Ordinary full forms instead use `NoOperationAttrs.INSTANCE`, so no negative axis sentinel
means "all axes." `MultiAxisReductionAttrs` carries a caller-ordered immutable set of distinct
normalized axes plus retention; `StatisticalReductionAttrs` adds non-negative correction.
`ArgExtremaAttrs` carries the normalized axis, retained-dimension choice, and an
explicit `ArgExtremaTiePolicy`. `MaskedReductionAttrs` carries only the normalized reduction axis for
masked, axis-removing `SUM` and `MEAN`. These reduction semantics are implemented, and public Tensor reduction
methods currently cover `sum`, `mean`, `prod`, reduction `min`, reduction `max`, boolean `all`,
boolean `any`, floating log-sum-exp/statistics/norms, and numeric `argMin`/`argMax`. Ordinary full
forms produce a canonical rank-zero scalar; single- and multi-axis forms normalize caller axes,
then remove them or retain them with extent one. Arg extrema
have no full form, accept floating and integral inputs, and produce fixed INT64 results.
Masked `sum(axis, mask)` and `mean(axis, mask)` require ordinary right-aligned broadcasting to
produce exactly the input Shape, remove the selected axis, and record exact `[input, mask]`
provenance.
Integral ordinary reduction and arg-extrema ordering/empty-domain semantics are documented below.
Numerical or truth evaluation, floating ordinary empty-domain behavior, gradients, and execution
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
`SliceKind.SLICE` and `SliceAttrs` define one signed-step slicing vocabulary. At each entry index,
the attributes pair one normalized inclusive start, selected length, normalized input axis, and
signed non-zero step. The semantic values contain no input Tensor, Shape, raw bound, or end
sentinel. Public `Tensor.slice` separately clones four equal-length request arrays, normalizes and
directionally clamps raw axes and bounds against selected static dimensions, derives same-rank
Shape and conditional layout metadata, and records normalized attributes plus one-input
provenance. Both `sliceAxis` overloads and `flip` use the same one-occurrence path. None reads
values, attaches storage, defines gradients, captures a graph, or executes work.
`PadKind.PAD` pairs with `PadAttrs`, whose immutable ordered `before` and `after` lists carry one
non-negative width per normalized axis position and whose `constantValue` retains one exact typed
scalar value. `TileKind.TILE` pairs with `TileAttrs`, whose immutable ordered list
carries one positive complete-pattern repeat count per normalized axis position. Both attributes
accept empty scalar-identity lists and `Long.MAX_VALUE` structurally. They contain no input Tensor
or Shape, so rank matching, checked result-Shape arithmetic, padding-constant conversion, result
descriptors, layout, provenance, gradients, materialization, compiler behavior, backend behavior,
ONNX mapping, and execution remain outside these semantic values. Public `Tensor.pad` and
`Tensor.tile` separately perform the input-dependent model validation and metadata construction
described under [pad and tile expressions](#pad-and-tile-expressions).
`UnaryElementwiseKind` names nineteen parameterless unary arithmetic, transcendental, and
activation meanings. `FloatingClassificationKind` separately names three parameterless floating
classifications with fixed BOOL results. `ScalarElementwiseKind` names eight parameterized
one-input meanings, with exact typed parameters carried by `ScalarValueAttrs` or
`ClampRangeAttrs`. The public `Tensor.add`, `sub`, `mul`, `div`, `minimum`,
`maximum`, and tensor-valued `pow` methods use the binary kinds to construct storage-free expression
tensors. The public `Tensor.greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`,
and `notEqualTo` methods use the comparison kinds to construct storage-free `BOOL` expressions
from ordered floating inputs. The public `Tensor.abs`, `neg`, `reciprocal`, `log`, `log1p`, `exp`,
`expm1`, `erf`, `sqrt`, `rsqrt`, `floor`, `ceil`, `sign`, `relu`, `sigmoid`, `tanh`, `gelu`,
`geluTanhApproximation`, and `silu` methods use the unary kinds to create storage-free expressions
from one floating input. `Tensor.isFinite`,
`isNaN`, and `isInf` use the classification kinds to create storage-free, non-differentiable BOOL
expressions from one floating input. The public scalar overloads
`Tensor.add(ScalarValue)`, `sub(ScalarValue)`, `mul(ScalarValue)`, `div(ScalarValue)`,
`minimum(ScalarValue)`, `maximum(ScalarValue)`, and `pow(ScalarValue)` use matching scalar kinds
and typed attributes to create one-input storage-free expressions with exact receiver/value type
matching. Their retained `double` overloads are exact-FLOAT64 adapters. First-class
`clamp(ScalarValue, ScalarValue)` retains one range operation; `clampMin` delegates to scalar
`maximum`, and `clampMax` delegates to scalar `minimum`. The public
`Tensor.logicalAnd`, `logicalOr`, and `logicalNot` methods use the logical kinds to construct
storage-free `BOOL` expressions from exact `BOOL` inputs. AND and OR derive a right-aligned
broadcast shape and preserve receiver/argument order; NOT retains the exact input `Shape`
reference without broadcasting. Every logical result has unresolved layout and false gradient
eligibility. Static `Tensor.where(condition, ifTrue, ifFalse)` requires an exact BOOL condition,
promotes the two floating branches, broadcasts the branches first and the condition with their
common shape second, and creates a fresh result whose gradient eligibility is the branch-only OR.
Its provenance preserves the exact ordered condition, true branch, and false branch references.
`Tensor.sum`, `prod`, reduction `min`, and reduction `max` construct full or single-axis floating
or integral aggregate expressions; `mean` remains floating-only. They preserve the input type and gradient-eligibility request,
derive the reduced shape, leave layout unresolved, and record the receiver as their sole
provenance input. Product and extrema preserve eligibility metadata without claiming that a
gradient rule exists.
`Tensor.all` and `Tensor.any` construct full or single-axis boolean aggregate expressions from exact BOOL
input. They use the same shape and provenance rules, produce exact BOOL with false gradient
eligibility, and do not inspect truth values or define empty-domain identities. Aggregate ALL and
ANY remain typed separately from elementwise AND and OR.
`Tensor.argMin` and `Tensor.argMax` construct single-axis index-producing expressions from floating
or integral input. Their convenience forms explicitly use `FIRST_INDEX`, while complete forms
retain an explicit first- or last-index policy. Every result is exact INT64 with false gradient
eligibility. The shared contract fixes floating/integral ordering and rejects a statically empty
selected axis; construction does not compare values or select an index.
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

### Numeric promotion

`DataTypePromotion` exposes two related contracts. `promoteFloating(left, right)` retains the
floating-only hierarchy:

```text
BFLOAT16 < FLOAT32 < FLOAT64
```

`promoteNumeric(left, right)` accepts a pair only when both operands belong to the floating
category above or both belong to the signed-integral hierarchy:

```text
INT32 < INT64
```

For integral pairs, INT64 wins when either operand is INT64; an INT32 operand then participates in
the signed INT64 domain by conceptual sign extension. Same-width pairs retain their type. BOOL,
null, and mixed floating/integral pairs are rejected. The implemented `Tensor.cast` method can
make a category conversion explicit, but it constructs expression metadata rather than converting
values; promotion itself never inserts a cast or evaluates a value.

For example:

```java
DataType result = DataTypePromotion.promoteFloating(
        DataType.BFLOAT16, DataType.FLOAT64);
```

The first input has 16 logical bits and the second has 64, so the widest precision is `FLOAT64`.
Similarly, `promoteNumeric(INT32, INT64)` returns `INT64`. These are model-level type decisions;
they do not prove evaluated values, compiler behavior, or backend support.

### BFLOAT16 representation

`BFloat16Bits` converts scalar values between Java `float` and raw BFLOAT16 bits held in a `short`. Conversion to BFLOAT16 uses round-to-nearest with ties to even, preserves signed zero and infinities, and canonicalizes NaN to `0x7FC0`.

The utility describes the value format only. It does not allocate storage, expose device formats, or report backend capabilities.

### Exact typed scalar values

`ScalarValue` is an immutable operation parameter consisting of one exact `DataType` and that
type's primitive bits. It is not a scalar Tensor: a rank-zero Tensor has a descriptor, identity,
and potentially storage, while `ScalarValue` owns none of those. It is also not a conversion
request or executable constant.

Named factories make the representation explicit:

```java
ScalarValue floatValue = ScalarValue.float32(0.5f);
ScalarValue rawBfloatNaN = ScalarValue.bfloat16Bits((short) 0x7FC1);
ScalarValue largeInteger = ScalarValue.int64(9_007_199_254_740_993L);
ScalarValue falseValue = ScalarValue.bool(false);
```

`float64` and `float32` retain raw IEEE-754 bits, including the sign of zero and distinct NaN
payloads. `bfloat16Bits` retains every supplied 16-bit pattern without canonicalizing NaN;
`bfloat16(float)` is the separately named binary32-to-BFLOAT16 conversion and uses
`BFloat16Bits.fromFloat`. `int32` and `int64` retain exact two's-complement values, so the INT64
example remains exact even though it is above `2^53`. `bool` stores only canonical false or true.
Equality and hashing use the data type and exact bits.

Each inspector is strict. For example, calling `float32Value()` on an INT64 value throws
`IllegalStateException` rather than narrowing or interpreting the bits. There is no general
conversion method. Scalar/clamp Tensor expressions require a matching floating value, while
constant padding accepts all six types when the value exactly matches the receiver:

```java
Tensor scaled = float32Input.mul(ScalarValue.float32(0.5f));
Tensor padded = boolInput.pad(before, after, ScalarValue.bool(false));
```

The existing primitive `TensorFactory.scalar`, `scalarBFloat16`, `full`, and `fullBFloat16`
methods are unchanged. They still create eager scalar or dense Tensor storage through their
existing exact primitive signatures; this API does not add a `ScalarValue` factory overload or
materialize a `ScalarValue` as a Tensor.

## Shapes and dimensions

The public shape contracts live in `io.github.pho001.synaptik.model.shape`. `Shape` is an immutable ordered collection of `Dimension` values. A dimension is one of:

- `StaticDimension`, with a known non-negative `long` size;
- `DynamicDimension`, with a canonical non-blank symbolic name; or
- `ExpressionDimension`, with an exact symbolic formula or an identity-based constrained unknown.

Every non-static form is dynamic: `isDynamic()` is true, `staticSize()` is empty, and only the
named `DynamicDimension` has a present `dynamicSymbol()`. Dynamic dimensions are explicit values
and never use negative numeric sentinels. Shape values defensively isolate caller-owned arrays
and expose immutable dimension lists.

### Symbolic extent expressions

`DimensionExpressions` is the field-free public construction boundary for derived extents. Its
six methods create checked addition, signed constant offset, multiplication by a non-negative
constant, floor or ceiling division by a positive constant, and a distinct constrained unknown:

```java
add(left, right)
addConstant(input, offset)
multiply(input, factor)
floorDivide(input, divisor)
ceilingDivide(input, divisor)
unknown(minimum, maximum)
```

Construction returns the simplest truthful `Dimension`. Fully static arithmetic returns a
`StaticDimension`. Adding zero, multiplying by one, or dividing by one returns the exact input
reference; multiplying by zero returns static zero. Linear sums flatten nested sums, fold static
terms into a checked signed offset, combine repeated dimensions, and ignore operand order for
equality. Thus `N + N` equals `2 * N`, and independently constructed `N + M` and `M + N` are
equal. Floor and ceiling division remain explicit structural nodes and are not reassociated with
addition or multiplication.

An `ExpressionDimension` exposes its read-only `DimensionExpression`. The four public forms are:

- `LinearCombination`, an immutable non-empty map of dimensions to positive `long`
  coefficients plus a signed `long` offset;
- `FloorDivision` and `CeilingDivision`, each with one dividend and a positive constant divisor;
  and
- `Unknown`, with an inclusive non-negative minimum and an optional inclusive maximum dimension.

Exact formula forms use structural equality. `Unknown` deliberately uses object identity, so two
separate calls with equal bounds remain different extents; reusing the same returned dimension
preserves that unknown's identity. The optional maximum retains the exact supplied dimension
reference. None of these values binds a symbol, evaluates a concrete size, solves graph-wide
equalities, or carries compiler, layout, storage, runtime, or backend state.

#### Complete symbolic-extent example

##### Goal and inputs

Represent exact formulas derived from named extents `N` and `M`, inspect one canonical sum, and
show that a bounded unknown remains distinct. This example constructs model values only.

```java
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpression;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.ExpressionDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.Optional;

public final class SymbolicExtentExample {
    public static void main(String[] args) {
        Dimension n = new DynamicDimension("N");
        Dimension m = new DynamicDimension("M");
        Dimension nPlusTwo = DimensionExpressions.addConstant(n, 2);
        Dimension canonicalSum = DimensionExpressions.add(
                nPlusTwo, DimensionExpressions.multiply(m, 2));
        Dimension sameSum = DimensionExpressions.add(
                DimensionExpressions.multiply(m, 2),
                DimensionExpressions.add(new DynamicDimension("N"), new StaticDimension(2)));
        Dimension halfRoundedUp = DimensionExpressions.ceilingDivide(nPlusTwo, 2);
        Dimension unknown = DimensionExpressions.unknown(1, Optional.of(nPlusTwo));

        DimensionExpression.LinearCombination inspected =
                (DimensionExpression.LinearCombination)
                        ((ExpressionDimension) canonicalSum).expression();
        Shape shape = Shape.ofDimensions(canonicalSum, halfRoundedUp, unknown);

        System.out.println(canonicalSum.equals(sameSum));
        System.out.println(inspected.offset());
        System.out.println(shape);
        System.out.println(unknown.equals(
                DimensionExpressions.unknown(1, Optional.of(nPlusTwo))));
    }
}
```

##### Result and interpretation

The example prints:

```text
true
2
Shape[2 * M + N + 2, ceilDiv((N + 2), 2), unknown(min=1, max=(N + 2))]
false
```

The first result demonstrates operand-order-independent structural equality. The offset is `2`,
and the shape retains exact formula and bound diagnostics. The final result demonstrates that a
new unknown-construction call is not equal to the original unknown. The text format is diagnostic,
not serialization, and the example does not bind `N` or `M`, calculate a concrete element count,
construct a Tensor expression, or execute anything.

##### Failures and useful variations

- A negative multiplication factor, non-positive divisor, negative unknown minimum, or static
  maximum below the minimum fails with `IllegalArgumentException`.
- Checked coefficient, offset, and fully static arithmetic overflow fails with
  `ArithmeticException`.
- A signed offset may retain a formula such as `N - 3`, but every eventual concrete binding must
  still satisfy the non-negative dimension invariant. A fully static negative result is rejected
  immediately.

### Scalars and empty tensors

`Shape.scalar()` is the canonical rank-0 scalar shape. It has no axes and its known element count is one.

Static size zero is valid. For example, `Shape.of(2, 0, 4)` represents an empty tensor and has a known element count of zero. Static dimensions use `long`; later storage implementations may impose narrower allocation limits without changing the logical shape.

### Static and dynamic element counts

`Shape.knownElementCount()` returns a present checked `long` value only when every dimension is static. Dynamic shapes return an empty optional. Non-zero multiplication overflow raises `ArithmeticException`; a fully static shape containing a zero dimension has count zero even when multiplying other dimensions would overflow.

`Shape.toLongArray()` copies the ordered static sizes and rejects dynamic shapes. Positive and negative axes are normalized through the shape, while every axis is invalid for a rank-0 scalar.

### Broadcasting

`ShapeBroadcast.broadcast(left, right)` applies right-aligned broadcasting. Equal dimensions are preserved and static size `1` expands to the opposing dimension. This includes scalar broadcasting and zero-sized dimensions such as `[0, 3]` with `[1, 3]`.

Equal named symbols and structurally equal exact expressions are compatible, and a singleton may
expand to any dynamic dimension. A constrained unknown is equal only when the exact same unknown
is reused. Different symbols, unequal expressions, distinct unknowns, or a dynamic dimension
paired with a non-singleton static size are rejected because local model code cannot prove their
compatibility. Graph-wide symbolic constraints belong to future compiler shape inference.

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

### Expression producers and indexed provenance

A derived Tensor is one indexed result of one `TensorProducer`. The producer is immutable
pre-capture expression-occurrence identity: it retains one exact `Operation`, an immutable ordered
snapshot of exact input Tensor references, and an immutable ordered snapshot of exact output
descriptor references. The selected operation signature validates those final input and output
counts. Separate producer objects remain separate occurrences even when all three parts are
structurally equal.

`TensorProvenance(producer, outputIndex)` associates one result Tensor with a zero-based producer
position. Its `operation()`, `inputs()`, and `outputDescriptor()` methods derive their exact
references from the producer. A derived Tensor must retain the same descriptor object as its
selected producer position; a merely equal descriptor object is rejected. Single-output
expressions use a one-descriptor producer and index zero.

The producer stores descriptors, not output Tensor objects. References therefore point only from
result Tensor to provenance to producer, so the model creates no producer/result cycle. Producer
identity is not a `NodeId`, `ValueId`, graph membership, compiler-capture result, runtime state, or
execution promise. The producer constructor and multi-output construction seam are package-private;
the public API currently exposes only the producers created by existing Tensor expressions.

#### Single-output example

##### Goal and inputs

Observe the producer and provenance of one current unary expression. The input is a storage-backed
`FLOAT32` Tensor with Shape `[2]`; the result is metadata for negation, not evaluated values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class SingleOutputProducerExample {
    public static void main(String[] args) {
        Tensor input = TensorFactory.zeros(
                Shape.of(2), DataType.FLOAT32, Optional.empty(), true);
        Tensor result = input.neg();

        TensorProvenance origin = result.provenance().orElseThrow();
        TensorProducer producer = origin.producer();

        System.out.println(producer.outputCount());
        System.out.println(origin.outputIndex());
        System.out.println(producer.inputs().getFirst() == input);
        System.out.println(origin.outputDescriptor() == result.descriptor());
        System.out.println(producer.operation() == origin.operation());
    }
}
```

##### Result and interpretation

The program prints:

```text
1
0
true
true
true
```

The result has one producer output at index zero, retains the exact input reference, and shares
the exact descriptor and operation references exposed through provenance. It does not prove that
negation ran, that a graph was captured, or that a gradient rule exists.

##### Useful variation

Calling `input.neg()` twice creates two structurally equivalent but identity-distinct producers.
The model does not intern them or perform common-subexpression elimination.

#### Conceptual two-output example

This is a conceptual model illustration, not a callable public multi-output API or an implemented
operation:

```text
producer P
  operation:          hypothetical two-output operation
  inputs:             [input]
  outputDescriptors: [primaryDescriptor, auxiliaryDescriptor]

primaryResult.provenance   = (P, 0)
auxiliaryResult.provenance = (P, 1)
```

Both results share the exact producer reference, while each retains the exact descriptor from its
own ordered position. `P` retains neither result Tensor. This illustrates what the model can
represent; no production multi-output operation, public sibling-result lookup, compiler capture,
or graph-node creation is implemented by this contract.

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
conversion, random generation, typed access/export, view preservation, or execution; integer
range creation is a separate factory method described below.

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

Range label and argument validation precede allocation. A successful call builds one complete
exact carrier and delegates once to matching flat import. A blank label therefore fails after the
carrier, destination, and ID have been allocated but before the flat copy; it consumes the ID.
Identifier exhaustion is also observed after both arrays exist and before copying. Neither failure
rolls back an identifier.

Strict and cyclic prefix preparation is not a production Tensor API. Repository tests may prepare
such arrays in test source and pass the finished carrier to public flat import, but no public
factory method or product inventory exposes that fixture convenience.

### Complete integer-range example

#### Goal and inputs

Create the INT32 range `[1, 4, 7]` from inclusive start `1`, exclusive end `8`, and step `3`.

```java
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class IntegerRangeExample {
    public static void main(String[] args) {
        Tensor range = TensorFactory.range(
                1, 8, 3, Optional.of("indices"));

        int[] rangeData = (int[]) range.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(range.descriptor().dataType());
        System.out.println(range.descriptor().shape());
        System.out.println(Arrays.toString(rangeData));
    }
}
```

#### Meaningful lines

- The `int` range overload fixes the result data type as `INT32`. Ceiling division of the distance
  by step gives three values, and the exclusive bound `8` is not emitted.
- The result shape is rank one with one position for each of those three values. Heap-base
  inspection only makes existing raw storage observable; it is not a typed Tensor access or
  export API.

#### Result and interpretation

The program prints:

```text
INT32
Shape[3]
[1, 4, 7]
```

The result demonstrates range typing, exclusive-end semantics, canonical dense shape, and copied
storage. It does not provide an empty or floating range, implicit conversion, a prefix fixture
API, typed access/export, provenance, or execution. Normal random creation is the separate
explicit-source `TensorRandoms` method described next.

#### Failures and useful variations

- `range(0, 3, -1, label)` fails because the negative step cannot advance toward the larger end.
- Equal start and end values fail because current range construction is deliberately non-empty.
- Counts above `Integer.MAX_VALUE` fail before carrier, destination, or ID allocation.

`TensorRandoms` owns exactly one public normal method:
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

The generator is transient caller-owned state. `TensorRandoms` does not select an algorithm or
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
import io.github.pho001.synaptik.model.tensor.TensorRandoms;
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
        Tensor tensor = TensorRandoms.randomNormal(
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
  `nextGaussian()` with four known values. `TensorRandoms` uses that exact object and does not call
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

`TensorRandoms` also owns exactly one public continuous-uniform method:
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
may round to a lower representable value. `TensorRandoms` does not clamp, resample, or post-validate a
custom non-conforming source result. A scalar consumes one bounded call; an empty shape consumes
none.

The source has the same ownership and reproducibility boundary as normal creation. The caller
creates, configures, seeds, owns, and advances it, and must provide exclusive or otherwise safe
access. `TensorRandoms` does not select an algorithm or default source, store a seed, retain or replace
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
import io.github.pho001.synaptik.model.tensor.TensorRandoms;
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
        Tensor tensor = TensorRandoms.randomUniform(
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

- The source overrides only the bounded `nextDouble(origin, bound)` path used by `TensorRandoms` and
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

`TensorRandoms` owns exactly two public bounded-integral overloads named `randomInt`. Primitive
`int` bounds produce `INT32`, while primitive `long` bounds produce `INT64`. Both take a fully
static Java-array-sized shape, inclusive origin, exclusive bound, caller-owned `RandomGenerator`,
and optional label. The origin must be strictly less than the bound, including for an empty
result. The synthesized descriptor is canonical dense-contiguous and always has
`requiresGrad == false`; there is no data-type or gradient parameter because the bound carrier
already selects a non-differentiable result type.

Each logical row-major element is the direct result of exactly one matching bounded call:
`nextInt(origin, bound)` for `INT32` or `nextLong(origin, bound)` for `INT64`. `TensorRandoms` uses no
unbounded draw, modulo reduction, floating arithmetic, narrowing, widening, or alternate carrier.
It fills one exact `int[]` or `long[]` and delegates once to matching flat import. A conforming
source supplies values in the half-open interval `[origin, bound)` without project-owned modulo
bias; custom non-conforming values are copied without post-validation. A scalar consumes one
bounded call and a valid empty result consumes none.

The source has the same ownership and bounded reproducibility contract as the floating random
methods. The caller configures, seeds, owns, advances, and provides safe access to it. `TensorRandoms`
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
import io.github.pho001.synaptik.model.tensor.TensorRandoms;
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
        Tensor tensor = TensorRandoms.randomInt(
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
  `nextInt(origin, bound)` invocation. `TensorRandoms` stores each returned value directly.
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

`TensorRandoms` owns exactly one public Bernoulli method:
`randomBernoulli(shape, probability, randomGenerator, label)`. It creates a canonical dense BOOL
tensor with gradients disabled. The shape must be fully static, its checked logical count must fit
a Java array, and the binary64 probability must be finite and in the closed interval `[0, 1]`.
Positive and negative zero are accepted as zero. No data-type or gradient argument is present
because Bernoulli output is logical non-differentiable leaf data; numeric truthiness and output
conversion are outside this method.

For each logical row-major element, `TensorRandoms` calls the exact supplied source's unbounded
`nextDouble()` method once and stores canonical byte `1` exactly when `draw < probability`;
otherwise it stores byte `0`. The strict comparison means that a draw equal to the probability is
false. A custom non-conforming draw is compared directly without post-validation. Calls are not
skipped at probability zero or one: conforming draws still produce all false or all true values,
respectively, while source advancement stays independent of endpoint optimization. A scalar
consumes one call and an empty result consumes none.

The source follows the same caller-ownership and bounded-reproducibility contract as the other
random methods. `TensorRandoms` does not substitute, retain, seed, synchronize, reset, split, or close
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
import io.github.pho001.synaptik.model.tensor.TensorRandoms;
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
        Tensor tensor = TensorRandoms.randomBernoulli(
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

- The source overrides unbounded `nextDouble()` and rejects `nextLong()`. `TensorRandoms` uses the
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

### Explicit graph RNG state

`GraphRngState` is the implemented public boundary for one explicit RNG state occurrence in a
Tensor expression graph. It is distinct from eager `TensorRandoms`: eager initialization consumes
a caller-owned JDK `RandomGenerator` immediately and returns host-backed leaf data, whereas graph
state records storage-free semantics for a future prepared implementation to consume.

The public initializer is `GraphRngState.initial(long key, long counter)`. Both arguments are raw
unsigned 64-bit words carried by Java `long`; every bit pattern is valid. Signed decimal display
and signed comparison do not define their meaning. The key identifies a caller-selected stream or
domain. The counter identifies the next abstract logical sample position. A future consuming
operation retains the key and advances the counter modulo `2^64` by that operation's documented
logical draw count.

Each call privately wraps one fresh, unlabeled, storage-free Tensor with exact descriptor
`INT64`, `Shape.of(2)`, unresolved layout, and `requiresGrad == false`. Its
`GraphRngKind.INITIAL_STATE` producer has `GraphRngStateAttrs(key, counter)`, no inputs, one output,
and provenance output index zero. Lane zero conceptually carries the key bits and lane one the
counter bits. The wrapper intentionally provides no public Tensor, lane, storage, mutation,
generator, split, or copy accessor, so those lanes are not an invitation to apply numerical Tensor
operations to state.

Instances use object-identity equality. Two calls with equal words request replay-equivalent
abstract stream positions but create distinct state, Tensor, producer, and identifier
occurrences. The wrapper is shallowly immutable and may be shared, but it performs no
synchronization. Branching one state to multiple future consumers deliberately reuses the same
abstract interval; sequential use threads the returned next state instead.

#### Complete current-state example

##### Goal and inputs

Create two expression occurrences at the same explicit abstract position. The key bits are
`0x1234`, and the counter bits are zero.

```java
import io.github.pho001.synaptik.model.tensor.GraphRngState;

public final class GraphRngStateExample {
    public static void main(String[] args) {
        GraphRngState start = GraphRngState.initial(0x1234L, 0L);
        GraphRngState replay = GraphRngState.initial(0x1234L, 0L);

        System.out.println(start == replay);
        System.out.println(start.equals(replay));
    }
}
```

##### Meaningful lines

- Both initializer calls retain the same raw key/counter attributes, so they request the same
  abstract stream position.
- Each call creates a fresh expression occurrence. Both reference and inherited object equality
  therefore report that the wrappers are distinct.

##### Result and interpretation

The program prints:

```text
false
false
```

This proves public construction and identity equality. It does not expose the private state
Tensor, sample a value, choose a random algorithm, or prove a cross-backend bitstream.

### Explicit-state dropout construction

`Tensor.dropout(double probability, GraphRngState state)` is the implemented model-construction
boundary for training-only inverted dropout. It accepts only a floating receiver and a finite drop
probability in `[0.0, 1.0)`. The result is a `DropoutResult(output, nextState)`; it deliberately has
no public mask component.

One call creates one exact producer with these ordered positions:

| Slot | Role | Descriptor | Publicly returned |
|---:|---|---|---|
| input 0 | input value | Exact input floating type and Shape | Existing receiver |
| input 1 | RNG state | Private `INT64 Shape[2]` state Tensor | Through opaque input state only |
| output 0 | dropped value | Input type/Shape/gradient eligibility; unresolved layout | `DropoutResult.output()` |
| output 1 | auxiliary keep mask | `BOOL`, exact input Shape, no gradients, unresolved layout | No |
| output 2 | next RNG state | `INT64 Shape[2]`, no gradients, unresolved layout | `DropoutResult.nextState()` |

The table shows producer roles, not storage. Every output is unlabeled and storage-free. The
multi-output factory creates three fresh Tensor wrappers and three Tensor IDs, one for each output
slot. The slot-one wrapper is discarded from the public result, while the producer retained by
slots zero and two still describes its BOOL mask position for later compiler capture.

For logical element position `i`, the operation means one abstract draw `u[i]` in `[0, 1)` and:

```text
keep[i] = u[i] >= probability
output[i] = keep[i] ? input[i] / (1 - probability) : +0
```

This is inverted dropout: kept values are scaled so their ideal expectation matches the input.
Dropped values are positive zero even for negative zero, NaN, or infinite input. Kept signed zero
and infinity retain their sign; kept NaN remains NaN without a payload promise. Construction does
not evaluate this formula or select its finite-precision algorithm.

#### Complete dropout-construction example

##### Goal and inputs

Build two replay-equivalent dropout occurrences and one explicitly threaded successor for a
storage-free `FLOAT32` input with Shape `[2, 3]`. The probability is `0.1`, the key is `0x1234`,
and the initial counter is zero.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.DropoutResult;
import io.github.pho001.synaptik.model.tensor.GraphRngState;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.Optional;

public final class DropoutConstructionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        Tensor activations = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));
        GraphRngState start = GraphRngState.initial(0x1234L, 0L);
        DropoutResult first = activations.dropout(0.1d, start);
        DropoutResult replay = activations.dropout(
                0.1d, GraphRngState.initial(0x1234L, 0L));
        DropoutResult second = first.output().dropout(0.1d, first.nextState());

        TensorProducer producer = first.output().provenance().orElseThrow().producer();
        System.out.println(producer.inputs().size());
        System.out.println(producer.outputCount());
        System.out.println(producer.outputDescriptors().get(1).dataType());
        System.out.println(first.output().descriptor().shape() == shape);
        System.out.println(first.output() == replay.output());
        System.out.println(second.output() == first.output());
    }
}
```

##### Meaningful lines

- `first` and `replay` use equal key/counter words, so eventual conforming execution requests the
  same abstract interval. They remain distinct Tensor and producer occurrences.
- `second` consumes `first.nextState()`, which expresses sequential non-overlap without mutating
  `start` or storing a hidden generator.
- The producer exposes two inputs and three descriptors. Descriptor slot one is the non-public
  BOOL auxiliary keep mask.

##### Result and interpretation

The program prints:

```text
2
3
BOOL
true
false
false
```

This proves the current public construction surface, output-slot description, Shape identity, and
fresh occurrence identity. It does not sample a mask, calculate output values, expose the state
Tensor, capture a graph, construct a gradient, or execute a backend.

#### State advancement, branching, and edge cases

For a bound logical element count `N`, eventual conforming execution consumes counter positions
`counter` through `counter + N - 1` and returns the same key with `counter + N modulo 2^64`.
Scalar input uses `N == 1`. A fully static Shape uses its mathematical extent product, including
products too large for signed `long`; a dynamic or symbolic Shape uses the non-negative count after
its extents are bound. Any bound zero extent gives `N == 0`, so output and mask are empty and the
next state has the same abstract key/counter value while remaining a distinct expression
occurrence.

Both signed probability-zero values are retained. Because every draw lies in `[0, 1)`, probability
zero keeps every value, but a non-empty occurrence still consumes one draw per element and advances
state. Reusing one state for two calls intentionally branches over the same interval. Threading
each returned `nextState` expresses sequential intervals. Sharing the immutable wrapper does not
synchronize execution.

Input eligibility is checked before probability, and probability before state nullity. Integral or
BOOL input fails; NaN, infinity, negative probability, and probability at least one fail; then a
null state fails. These local failures allocate no Tensor ID. Successful construction performs no
sampling, reads no storage, and mutates neither input nor state.

The raw words, fixed descriptor, occurrence ordering, and modular advancement meaning are portable
model semantics. No
portable pseudorandom-number-generator algorithm, key schedule, counter-to-bits function,
floating conversion, or bitstream is selected, so equal state does not promise bitwise samples
across backends, routes, providers, or versions. Until an algorithm is selected, sampled replay is
bounded to the same conforming prepared implementation and configuration. A future graph
serializer must preserve both raw words losslessly, but no byte encoding, parser, schema version,
or stable enum token exists today. Compiler capture and serialization policy, prepared state,
materialization, sampling, kernels, runtime execution, gradients, and backend support remain in
their owning future layers.

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

Every public `TensorFactory` and `TensorRandoms` creation method returns a provenance-free leaf. The
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
| `minimum` | pairwise minimum of the left and right operands |
| `maximum` | pairwise maximum of the left and right operands |
| `pow` | left base raised to the right exponent |

The receiver is always the ordered left input and the argument is always the ordered right input.
This order remains in provenance even for operations whose mathematical result is commonly
commutative.

Pairwise `minimum` and `maximum` propagate NaN, order infinities normally, and choose negative
zero for minimum or positive zero for maximum when comparing opposite signed zeros. They promise
no NaN payload or bitwise result. The reduction names remain `min` and `max`.

ADD, SUB, MUL, MIN, and MAX accept either two floating operands or two signed-integral operands.
Floating pairs retain `BFLOAT16 < FLOAT32 < FLOAT64`; integral pairs use `INT32 < INT64`, with an
INT32 operand conceptually sign-extended into the promoted INT64 domain. DIV and POW remain
floating-only. BOOL and mixed floating/integral pairs are rejected; callers use an explicit
`cast` when cross-category conversion is intended. Every accepted method applies the existing
right-aligned local broadcast rule and creates a fresh `TensorDescriptor`. The result layout is
unresolved even when every shape
dimension is static because expression construction has not chosen storage geometry. Result
`requiresGrad` is the logical OR of the input requests; integral descriptors necessarily have it
false. That flag records gradient eligibility only and does not install a gradient rule.

Integral ADD, SUB, and MUL mean fixed-width two's-complement modular arithmetic in the promoted
result type: INT32 wraps modulo 2^32 and INT64 wraps modulo 2^64, reinterpreted as signed. Integral
MIN and MAX use ordinary signed order. Construction records these requests but neither reads
values nor detects overflow.

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
- Integral ADD, SUB, MUL, MIN, and MAX are valid. Integral DIV or POW fails before broadcasting
  and result-ID allocation. BOOL and mixed floating/integral pairs fail through numeric promotion;
  no implicit cast is inserted.
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

Each method accepts every same-category ordered pair from the floating or signed-integral types.
The common comparison domain uses the unchanged floating hierarchy or `INT32 < INT64`; an INT32
operand is conceptually sign-extended when the integral domain promotes to INT64. BOOL and mixed
floating/integral pairs are rejected unless the caller first records an explicit cast. The
promoted type is not the output type. Right-aligned local broadcasting derives the result shape.
The result data
type is always `BOOL`, its layout is unresolved, and `requiresGrad` is always false even when one
or both inputs request gradients.

Every valid call returns a fresh Tensor with a new factory identity, no label, and no host storage.
Its provenance contains one `Operation` with the exact matching `BinaryComparisonKind` and
`NoOperationAttrs.INSTANCE`, followed by exact ordered input references `[left, right]`. Repeated,
self, and symmetric calls are not interned, reordered, or canonicalized. Integral extrema and all
six relations use ordinary signed order; EQUAL and NOT_EQUAL compare exact promoted signed values.
Floating numerical edge policy, compiler capture, training-graph treatment, gradient rules, and
backend execution remain later responsibilities.

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
- Every INT32/INT64 ordered pair is valid. BOOL and mixed floating/integral pairs fail numeric
  validation before broadcasting and result-ID allocation; no implicit cast is inserted.
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

### Unary numeric transforms and floating classifications

Nineteen zero-argument unary methods create one-input, floating-preserving elementwise semantics:

| Method | Elementwise meaning |
|---|---|
| `abs` | Absolute magnitude. |
| `neg` | Additive inverse. |
| `reciprocal` | Multiplicative reciprocal. |
| `log` | Natural logarithm. |
| `log1p` | Natural logarithm of one plus the input. |
| `exp` | Portable natural exponential request. |
| `expm1` | Natural exponential of the input minus one. |
| `erf` | Gaussian error function. |
| `sqrt` | Principal square root. |
| `rsqrt` | Reciprocal of the principal square root. |
| `floor` | Greatest integer-valued result not greater than the input. |
| `ceil` | Least integer-valued result not less than the input. |
| `sign` | Numeric negative, zero, or positive classification. |
| `relu` | Rectified linear unit. |
| `sigmoid` | Logistic sigmoid. |
| `tanh` | Portable hyperbolic tangent request. |
| `gelu` | Exact Gaussian error linear unit, `x * Phi(x)`. |
| `geluTanhApproximation` | Fixed conventional tanh approximation to GELU. |
| `silu` | Sigmoid linear unit, `x * sigmoid(x)`. |

Each method accepts only `BFLOAT16`, `FLOAT32`, or `FLOAT64`. The result retains the exact input
data type, immutable `Shape` reference, and `requiresGrad` flag, while leaving layout unresolved.
That flag is eligibility metadata; it does not assert that a derivative or backward rule exists.

`rsqrt`, `log1p`, `expm1`, both GELU variants, and SiLU are first-class transforms, not stored
compositions. Exact GELU selects
`x * Phi(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`. The explicitly named tanh approximation selects
the fixed function
`0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`; it is not configurable permission
to substitute another approximation. SiLU selects `x * sigmoid(x) = x / (1 + exp(-x))`. The API
uses canonical `silu` naming and provides no `swish` alias. Their selected special-value meanings
are:

| Method | Selected special-value behavior |
|---|---|
| `rsqrt` | Positive/negative zero becomes same-signed infinity; positive infinity becomes positive zero; negative finite values and negative infinity produce NaN; NaN produces NaN. |
| `log1p` | Signed zero is preserved; `-1` produces negative infinity; values below `-1`, including negative infinity, produce NaN; positive infinity remains positive infinity; NaN produces NaN. |
| `expm1` | Signed zero is preserved; negative infinity becomes exactly `-1`; positive infinity remains positive infinity; NaN produces NaN. |
| `gelu` | Negative infinity becomes negative zero by continuous extension; signed zero is preserved; positive infinity remains positive infinity; NaN produces NaN. |
| `geluTanhApproximation` | Negative infinity becomes negative zero by continuous extension; signed zero is preserved; positive infinity remains positive infinity; NaN produces NaN. |
| `silu` | Negative infinity becomes negative zero by continuous extension; signed zero is preserved; positive infinity remains positive infinity; NaN produces NaN. |

These names select portable mathematical targets. The infinity entries for GELU and SiLU are
function-level continuous extensions, not a required literal order of primitive floating
operations. The names do not promise correct rounding, a fixed
unit-in-the-last-place (ULP) or relative-error bound, a bitwise result, an algorithm, or a backend
route. Compiler and backend work may preserve useful near-zero accuracy for `log1p` and `expm1`
because the model records the transforms without decomposition.

Three separate floating-classification methods describe BOOL values:

| Method | True exactly when |
|---|---|
| `isFinite` | The input is finite: normal, subnormal, positive zero, or negative zero. |
| `isNaN` | The input is NaN, independent of sign, quiet/signaling encoding, or payload. |
| `isInf` | The input is positive or negative infinity. |

Classification accepts the same three floating input types but always produces `BOOL`, retains
the exact input `Shape`, leaves layout unresolved, and sets `requiresGrad=false`. These methods
are non-differentiable value classifications, not numeric unary transforms, trace diagnostics,
eager Java booleans, or validation checks. For every represented floating value, exactly one of
the three classifications is true when the operation is eventually evaluated.

Every valid transform or classification call returns a fresh Tensor with no label or host
storage. Its provenance contains one parameterless `Operation`, exactly the receiver reference,
one producer output, and output index zero. Model construction reads no values and calculates no
result.

#### Complete transform-and-classification example

##### Goal and inputs

Build one exact `GELU` transform and one `IS_NAN` classification from a storage-free `FLOAT32`
Tensor of Shape `[2, 3]`. The example observes metadata only; it does not claim transformed or
classified element values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class FloatingMetadataExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));

        Tensor transformed = input.gelu();
        Tensor classified = input.isNaN();
        TensorProvenance transformOrigin = transformed.provenance().orElseThrow();
        TensorProvenance classificationOrigin = classified.provenance().orElseThrow();

        System.out.println("transformType=" + transformed.descriptor().dataType());
        System.out.println("transformGrad=" + transformed.descriptor().requiresGrad());
        System.out.println("transformKind=" + transformOrigin.operation().kind());
        System.out.println("classificationType=" + classified.descriptor().dataType());
        System.out.println("classificationGrad=" + classified.descriptor().requiresGrad());
        System.out.println("classificationKind=" + classificationOrigin.operation().kind());
        System.out.println("sameShape="
                + (transformed.descriptor().shape() == shape
                && classified.descriptor().shape() == shape));
        System.out.println("parameterless="
                + (transformOrigin.operation().attrs() == NoOperationAttrs.INSTANCE
                && classificationOrigin.operation().attrs() == NoOperationAttrs.INSTANCE));
        System.out.println("exactInputs="
                + (transformOrigin.inputs().getFirst() == input
                && classificationOrigin.inputs().getFirst() == input));
        System.out.println("oneInputEach="
                + (transformOrigin.inputs().size() == 1
                && classificationOrigin.inputs().size() == 1));
        System.out.println("outputIndexes="
                + transformOrigin.outputIndex() + "," + classificationOrigin.outputIndex());
    }
}
```

##### Meaningful lines and intermediate results

- `input.gelu()` records one `UnaryElementwiseKind.GELU` occurrence and preserves the input's
  floating type and gradient-eligibility request.
- `input.isNaN()` records one distinct `FloatingClassificationKind.IS_NAN` occurrence and fixes
  the result descriptor to non-gradient `BOOL`.
- Both results retain the exact `shape` and input references, use canonical no-attributes state,
  and select output index zero from a one-output producer.

##### Result and interpretation

The program prints:

```text
transformType=FLOAT32
transformGrad=true
transformKind=GELU
classificationType=BOOL
classificationGrad=false
classificationKind=IS_NAN
sameShape=true
parameterless=true
exactInputs=true
oneInputEach=true
outputIndexes=0,0
```

The output proves the result-type distinction, exact operation kinds, Shape and input retention,
parameterless construction, and one-input, output-index-zero provenance. It does not prove that
GELU or NaN classification ran, that numerical accuracy or a gradient rule exists, that a graph
was captured, or that any backend supports either operation.

##### Failures and useful variations

- Calling any of these methods on an `INT32`, `INT64`, or `BOOL` Tensor fails with
  `IllegalArgumentException` before result identity allocation; no implicit cast is inserted.
- Scalar, zero-sized, ordinary static, and dynamic Shapes are accepted. A resolved input layout is
  deliberately not copied to either result.
- Stored values, including signed zero, infinity, and NaN, are not inspected during construction.
- Chaining a transform records the immediately preceding result as its exact input.
- Exhausting the factory's Tensor-ID space fails after local descriptor and provenance values are
  built.

### Scalar arithmetic and clamp expressions

The scalar arithmetic surface parallels the seven Tensor-to-Tensor relationships. Each has one
exact `ScalarValue` overload and one retained exact-FLOAT64 `double` convenience:

| Typed method | FLOAT64 convenience | Elementwise meaning | Attributes |
|---|---|---|---|
| `add(ScalarValue)` | `add(double)` | Add the scalar. | `ScalarValueAttrs` |
| `sub(ScalarValue)` | `sub(double)` | Subtract the scalar from the input. | `ScalarValueAttrs` |
| `mul(ScalarValue)` | `mul(double)` | Multiply by the scalar. | `ScalarValueAttrs` |
| `div(ScalarValue)` | `div(double)` | Divide the input by the scalar. | `ScalarValueAttrs` |
| `minimum(ScalarValue)` | `minimum(double)` | Select the pairwise scalar minimum. | `ScalarValueAttrs` |
| `maximum(ScalarValue)` | `maximum(double)` | Select the pairwise scalar maximum. | `ScalarValueAttrs` |
| `pow(ScalarValue)` | `pow(double)` | Raise the input to the scalar exponent. | `ScalarValueAttrs` |

The clamp surface retains one first-class range operation and two one-bound conveniences:

| Typed method | FLOAT64 convenience | Meaning | Stored kind/attributes |
|---|---|---|---|
| `clamp(ScalarValue, ScalarValue)` | `clamp(double, double)` | Constrain to inclusive bounds. | `CLAMP` + `ClampRangeAttrs` |
| `clampMin(ScalarValue)` | `clampMin(double)` | Apply an inclusive lower bound through scalar maximum. | `MAX` + `ScalarValueAttrs` |
| `clampMax(ScalarValue)` | `clampMax(double)` | Apply an inclusive upper bound through scalar minimum. | `MIN` + `ScalarValueAttrs` |

Typed ADD, SUB, MUL, MIN, and MAX plus `clampMin` and `clampMax` accept floating or signed-integral
receivers. Typed DIV, POW, and first-class two-bound CLAMP remain floating-only. Every typed value
must have the receiver's exact data type; scalar attributes never promote. The `double` methods
construct exact FLOAT64 values; they do not
infer or narrow to another receiver type. Thus `float32Input.mul(0.5d)` fails, while
`float32Input.mul(ScalarValue.float32(0.5f))` is valid. The result retains the exact input data
type and immutable `Shape` reference, preserves `requiresGrad`, and leaves layout unresolved.
Integral results necessarily have false gradient eligibility.

Every valid call returns a fresh Tensor with a new factory identity, no label, and no host storage.
Its provenance contains the exact matching `ScalarElementwiseKind`, typed attributes, and exactly
the receiver reference. `clamp(minValue, maxValue)` is one first-class `CLAMP` operation with the
value meaning `minimum(maximum(input, lower), upper)`, but no stored intermediate producers.
`clampMin` creates exactly one scalar `MAX` producer, and `clampMax` exactly one scalar `MIN`
producer. Repeated, nested, identity-like, and special-value calls are not interned, folded, or
simplified at this boundary.

Floating scalar and Tensor-to-Tensor extrema share the same selected semantics: NaN propagates,
infinities
use normal numerical order, minimum selects negative zero from opposite signed zeros, and maximum
selects positive zero. The model promises no NaN payload, gradient convention, algorithm, backend
route, or execution. Integral MIN/MAX use ordinary signed order, and integral scalar ADD/SUB/MUL
use fixed-width two's-complement modular semantics. Expression construction records but does not
evaluate those meanings.

For `clamp`, null checks and floating-input eligibility precede bound construction. The bounds
must share one non-BOOL type, and `ClampRangeAttrs` compares the exact represented primitive type.
It rejects only a strict inversion; equal bounds, either ordering of signed zeros, ordered
infinities, and a floating NaN endpoint remain representable. The common bound type must then
equal the receiver type. NaN payload selection, numerical evaluation, optimization, gradients,
graph capture, and backend execution remain later responsibilities.

#### Complete scalar-expression example

##### Goal and inputs

Build a scalar multiplication followed by one range-clamp expression from a storage-free
`FLOAT32` tensor of shape `[2, 3]`. The example uses negative zero and a raw NaN payload to
demonstrate exact binary32 retention and observes expression metadata rather than numerical
values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
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

        ScalarValue negativeZero = ScalarValue.float32(-0.0f);
        ScalarValue payloadNaN = ScalarValue.float32(
                Float.intBitsToFloat(0x7FC0_0042));
        Tensor scaled = input.mul(negativeZero);
        Tensor result = scaled.clamp(negativeZero, payloadNaN);
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
                + Integer.toHexString(
                        Float.floatToRawIntBits(multiplier.value().float32Value())));
        System.out.println("clampKind=" + clampedFrom.operation().kind());
        System.out.println("minBits="
                + Integer.toHexString(
                        Float.floatToRawIntBits(range.minValue().float32Value())));
        System.out.println("maxBits="
                + Integer.toHexString(
                        Float.floatToRawIntBits(range.maxValue().float32Value())));
        System.out.println("exactInput=" + (clampedFrom.inputs().getFirst() == scaled));
        System.out.println("oneClampInput=" + (clampedFrom.inputs().size() == 1));
        System.out.println("fresh=" + (result != scaled && scaled != input));
    }
}
```

##### Meaningful lines and intermediate results

- `input.mul(negativeZero)` records scalar `MUL` and retains the exact `ScalarValue` reference and
  binary32 negative-zero pattern in one `ScalarValueAttrs`.
- `scaled.clamp(negativeZero, payloadNaN)` records exactly one `CLAMP` with both exact typed values
  in one `ClampRangeAttrs`. Its sole input
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
multiplierBits=80000000
clampKind=CLAMP
minBits=80000000
maxBits=7fc00042
exactInput=true
oneClampInput=true
fresh=true
```

The output proves exact attribute-bit retention, first-class range-clamp identity, descriptor
retention, fresh identity, and one-input provenance. It does not prove multiplication or clamp
values, parameter conversion for a backend, gradient rules, graph capture, backend support, or
execution.

##### Failures and useful variations

- Exact typed integral ADD, SUB, MUL, MIN, MAX, `clampMin`, and `clampMax` are valid. Integral DIV,
  POW, and range CLAMP fail before result identity allocation; BOOL remains invalid.
- `clamp(ScalarValue.float32(2.0f), ScalarValue.float32(1.0f))` fails with
  `IllegalArgumentException` and message
  `minValue must be less than or equal to maxValue`. On a non-floating input, the data-type failure
  occurs before that range failure.
- Signed zeros, infinities, and NaN payload bits remain unchanged in attributes. A NaN clamp
  endpoint is accepted because primitive `>` is false when either operand is NaN.
- A `double` convenience used on FLOAT32, BFLOAT16, INT32, or INT64 fails exact data-type matching;
  callers must construct the matching typed value explicitly.
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

All five families accept `BFLOAT16`, `FLOAT32`, or `FLOAT64`. Sum, product, minimum, and maximum
also accept `INT32` and `INT64`; mean remains floating-only. Every family preserves the exact input
data type without promotion or widening. A full form reduces every input axis to the canonical
rank-zero `Shape.scalar()` and records
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

The aggregate `min()` and `max()` families are distinct from pairwise elementwise
`minimum(Tensor)` and `maximum(Tensor)`. An aggregate form has one provenance input, uses
`AggregateReductionKind.MIN` or `AggregateReductionKind.MAX`, and reduces every axis or one
selected axis. A binary form has two
ordered provenance inputs, uses `BinaryArithmeticKind.MIN` or `BinaryArithmeticKind.MAX`, and
broadcasts corresponding elements without reducing rank. The distinct method names make pairwise
selection and aggregation visible at the call site; equal enum constant names do not make the
typed operation kinds equal.

For integral input, sum and product use fixed-width arithmetic modulo `2^32` or `2^64` in the exact
result type. Reassociation is permitted, so later execution may regroup operations but must include
every selected value exactly once. Integral minimum and maximum use signed order. Empty integral
domains have bounded identities:

| Kind | Empty INT32 result | Empty INT64 result |
|---|---:|---:|
| `SUM` | `0` | `0L` |
| `PROD` | `1` | `1L` |
| `MIN` | `Integer.MAX_VALUE` | `Long.MAX_VALUE` |
| `MAX` | `Integer.MIN_VALUE` | `Long.MIN_VALUE` |

The identity applies to a full empty input and independently to every empty selected-axis slice.
Retaining the axis does not change it. A different zero output axis may make the result itself
empty, leaving no output position at which to materialize an identity. Dynamic selected extents
use the same rule when later bound to zero. Floating behavior, masked reduction behavior, and BOOL
aggregate behavior remain unchanged. Expression construction records these meanings without
inspecting counts or values, implementing accumulation or comparison, creating gradient rules,
capturing a graph, lowering, reporting backend support, or executing.

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

- Sum, product, minimum, and maximum accept `INT32` and `INT64` without promotion. Mean rejects
  integral and BOOL input, while every numeric family rejects BOOL. No cast is inserted, and type
  validation precedes axis validation.
- An axis outside `[-rank, rank - 1]` fails with `IndexOutOfBoundsException`. Every axis is invalid
  for a scalar input, while a scalar full reduction is valid and still returns a fresh scalar.
- A rank-one axis removal returns `Shape.scalar()`. A zero extent or dynamic dimension does not
  trigger numerical inspection at this boundary.
- Repeated and nested calls always create fresh expression tensors; this API performs no
  canonicalization or common-subexpression elimination.

### Multi-axis and statistical reduction expressions

Twenty-six methods extend aggregate construction without changing full or single-axis overloads:

| Family | Axes removed | Axes optionally retained | Default correction |
|---|---|---|---:|
| `sum`, `mean`, `prod`, `min`, `max`, `all`, `any` | `family(int... axes)` | `family(int[] axes, boolean keepDimensions)` | — |
| `logSumExp`, `l1Norm`, `l2Norm` | `family(int... axes)` | `family(int[] axes, boolean keepDimensions)` | — |
| `variance`, `standardDeviation` | `family(int... axes)` | `family(int[] axes, boolean keepDimensions)` | `0` |
| Corrected variance/standard deviation | — | `family(int[] axes, boolean keepDimensions, long correction)` | Caller value |

Each positive or negative caller axis is normalized exactly once in caller order. The normalized
axes must be distinct. Order is retained in immutable attributes for equality, diagnostics,
transformation, and interchange, but Shape derivation uses axis membership and the numerical target
does not define a sequential fold. Reversing `[2, 0]` to `[0, 2]` therefore preserves the same
domain and Shape while producing a distinct attributes value. Caller arrays are cloned.

With `keepDimensions=false`, every selected axis is removed and each unselected immutable
`Dimension` reference is preserved in input order. With `true`, selected positions become new
static extent-one Dimensions. Removing every axis produces canonical `Shape.scalar()`. An empty
axis list is not full reduction: it selects a one-value point domain at every input position,
leaves Shape unchanged, and creates a fresh occurrence. Ordinary families and log-sum-exp return
the point value, norms return absolute value, and population variance/standard deviation return
positive zero. Correction at least one is invalid because point-domain count is one.

Ordinary input domains remain unchanged: SUM/PROD/MIN/MAX accept floating or signed integral,
MEAN is floating-only, and ALL/ANY are BOOL-only. Log-sum-exp, variance, standard deviation, and
both norms accept only floating input. Every result preserves exact input data type and gradient
eligibility, leaves layout unresolved, has no label/storage, and records a fresh one-output
producer with exact input `[input]` and provenance index zero. Construction does not read values,
create a gradient, capture a graph, lower, report backend support, or execute.

The portable floating target is exact real arithmetic plus the rules below, rounded once to result
format with round-to-nearest, ties-to-even. Equal-or-wider intermediates and reassociation are
permitted only within future conformance tolerance and exact special rules; narrower accumulation,
model-visible promotion, saturation, and bitwise guarantees are not selected. Finite overflow is
signed infinity. NaN payload/sign and signaling preservation are unspecified.

- SUM propagates NaN; opposite infinities produce NaN; empty is positive zero. Exact non-empty
  zero is negative only when every selected value is negative zero.
- MEAN is exact sum divided by positive count; empty and opposite infinities produce NaN; zero sign
  follows SUM.
- PROD propagates NaN; zero times infinity produces NaN; empty is positive one; zero/infinity sign
  follows multiplication parity.
- MIN/MAX propagate NaN, order infinities normally, select negative/positive zero respectively,
  and use positive/negative infinity for empty floating domains.
- Empty ALL is true and empty ANY is false.
- LOG_SUM_EXP means `log(sum(exp(x_i)))`: empty and all-negative-infinity domains produce negative
  infinity; NaN produces NaN; positive infinity wins unless NaN exists; a finite point preserves
  its value and signed zero.
- VARIANCE means `sum((x_i - mean)^2) / (N - correction)` and STANDARD_DEVIATION its non-negative
  principal square root. `N` must exceed correction. NaN or infinity produces NaN; a valid constant
  finite domain produces positive zero.
- L1 norm is `sum(abs(x_i))`; L2 norm is `sqrt(sum(x_i*x_i))`. Empty is positive zero, point is
  absolute value, NaN produces NaN, and infinity produces positive infinity unless NaN exists.

Integral ordinary reductions retain task 0018U1's exact-width modular sum/product, signed extrema,
and bounded empty identities. No algorithm, pass count, compensation scheme, traversal, vector
route, or kernel is part of these semantic contracts.

For corrected statistics, construction rejects negative correction and rejects a statically known
selected-domain count `N <= correction` before Tensor identity allocation. Static zero proves
`N=0`; checked non-zero multiplication overflow proves `N` exceeds every non-negative `long`
correction. Dynamic/expression counts are accepted structurally and must be proven valid by a later
owning compiler or binding path before execution.

#### Complete multi-axis metadata example

The following Java 26 example constructs metadata only:

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

Tensor input = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(2, 3, 4), Optional.empty(), true));

int[] axes = {2, 0};
Tensor totals = input.sum(axes, false);
Tensor retained = input.mean(new int[] {-1, 0}, true);
Tensor sampleVariance = input.variance(new int[] {2, 0}, false, 1);

axes[0] = 1; // does not change already constructed attributes

MultiAxisReductionAttrs totalAttrs = (MultiAxisReductionAttrs)
        totals.provenance().orElseThrow().operation().attrs();
StatisticalReductionAttrs varianceAttrs = (StatisticalReductionAttrs)
        sampleVariance.provenance().orElseThrow().operation().attrs();

System.out.println(totals.descriptor().shape());
System.out.println(retained.descriptor().shape());
System.out.println(totalAttrs.axes());
System.out.println(varianceAttrs.correction());
System.out.println(sampleVariance.provenance().orElseThrow().outputIndex());
```

Output:

```text
Shape[3]
Shape[1, 3, 1]
[2, 0]
1
0
```

This proves ordered normalization, caller-array isolation, structural Shape derivation, correction
retention, and index-zero provenance. It does not evaluate a sum, mean, or variance and does not
prove a numerical algorithm, gradient, compiler capture, backend route, or execution.

### Masked sum and mean expressions

The two masked aggregate overloads build one axis-removing expression from a floating input and
an exact BOOL mask:

| Public form | Operation kind | Result axis |
|---|---|---|
| `sum(axis, mask)` | `AggregateReductionKind.SUM` | Removed |
| `mean(axis, mask)` | `AggregateReductionKind.MEAN` | Removed |

The mask uses the same ordinary right-aligned broadcasting rule as other model expressions, but
the relationship is directional: `broadcast(inputShape, maskShape)` must equal `inputShape`.
The mask may use trailing dimensions and singleton expansion, but it may not add a leading axis
or enlarge an input singleton. A scalar mask broadcasts to the complete input. Equal symbolic or
expression dimensions are accepted when the existing local Shape rule can prove equality.

Right alignment does not infer semantic axis intent. For input `[batch, time, features]`, a mask
shaped `[batch, time]` generally compares `time` with `features` and is rejected. A caller that
means the first two axes inserts a trailing singleton explicitly, producing
`[batch, time, 1]`. That rank-edit expression remains visible as the exact mask producer consumed
by the reduction.

Every valid call returns a fresh Tensor with the input's exact floating data type and gradient
eligibility, unresolved layout, no label, and no host storage. Its operation carries
`MaskedReductionAttrs(normalizedAxis)`, and provenance retains exact ordered references
`[input, alignedMask]`. The mask's gradient eligibility does not affect the result.

The semantic meaning excludes false mask positions. Masked sum produces zero when no value is
selected. Masked mean divides each output by its selected true-count and produces NaN when that
count is zero; no NaN payload or bit pattern is promised. False positions exclude their input
before aggregation, including NaN and infinity. A static zero-sized reduction axis therefore
produces zero for every masked-sum slice and NaN for every masked-mean slice. A runtime zero-sized
or all-false dynamic slice follows the same semantic rule. Expression construction records these
rules but does not align storage, materialize a mask, inspect values, count positions, aggregate,
divide, create a gradient rule, capture a graph, or execute work.

#### Complete masked-reduction example

##### Goal and inputs

Build masked sum and mean metadata for a storage-free `FLOAT32` input shaped
`[batch, time, 4]` and a BOOL mask initially shaped `[batch, time]`. The example makes the mask's
intended axes explicit with `expandDims(2)`, then observes both producer boundaries and the
masked-reduction result metadata. It does not calculate aggregate values.

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

        Tensor alignedMask = mask.expandDims(2);
        Tensor sum = input.sum(1, alignedMask);
        Tensor mean = input.mean(-2, alignedMask);
        MaskedReductionAttrs attrs =
                (MaskedReductionAttrs) sum.provenance().orElseThrow().operation().attrs();

        System.out.println("alignedMaskShape=" + alignedMask.descriptor().shape());
        System.out.println("sumShape=" + sum.descriptor().shape());
        System.out.println("meanShape=" + mean.descriptor().shape());
        System.out.println("axis=" + attrs.axis());
        System.out.println("rankEditInput="
                + (alignedMask.provenance().orElseThrow().inputs().get(0) == mask));
        System.out.println("orderedInputs="
                + (sum.provenance().orElseThrow().inputs().get(0) == input
                && sum.provenance().orElseThrow().inputs().get(1) == alignedMask));
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

- `expandDims(2)` inserts a trailing singleton, changing the mask Shape from `[batch, time]` to
  `[batch, time, 1]`. The new Tensor records the original mask as its rank-edit producer input.
- Ordinary right alignment now compares `batch` with `batch`, `time` with `time`, and singleton
  `1` with feature extent `4`, so the mask broadcasts exactly to the input Shape.
- Both `1` and `-2` normalize to input axis `1`, the `time` axis.
- Removing input axis `1` produces `[batch, 4]`. The exact `batch` dimension object is retained.
- The masked operation's provenance stores the input first and the exact `alignedMask` Tensor
  second. Only the input supplies result type and gradient eligibility.

##### Result and interpretation

The program prints:

```text
alignedMaskShape=Shape[batch, time, 1]
sumShape=Shape[batch, 4]
meanShape=Shape[batch, 4]
axis=1
rankEditInput=true
orderedInputs=true
metadata=true
```

The output proves the caller-visible rank edit, ordinary broadcast-compatible Shape, local axis
normalization, axis removal, result metadata, and the two producer boundaries. It does not prove
mask materialization, selected values, a sum or mean result, gradient support, compiler capture,
backend support, or execution.

##### Failures and useful variations

- A null mask fails with `NullPointerException`. Integral or BOOL input and every non-BOOL mask
  fail with `IllegalArgumentException`; type checks occur before axis validation.
- An invalid axis fails with `IndexOutOfBoundsException`, including every axis for a scalar input.
  Axis validation occurs before mask broadcasting.
- Incompatible aligned dimensions fail with `IllegalArgumentException`. A compatible broadcast
  that would enlarge the input or add a leading result axis also fails. No cast, symbolic
  constraint, reshape, or rank edit is inserted.
- Scalar masks, equal-rank masks, lower-rank trailing masks, zero extents, equal dynamic symbols,
  exact equal expressions, reused unknown dimensions, and mask-side singleton dimensions are
  supported structurally. Distinct unknown dimensions remain locally unprovable. Repeated valid
  calls remain fresh expressions.

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

### Arg-extrema expressions

The six current `argMin` and `argMax` methods build index-producing aggregate metadata along
exactly one input axis:

| Public forms | Retained dimension | Tie policy |
|---|---|---|
| `argMin(axis)`, `argMax(axis)` | Removed | Explicitly supplies `FIRST_INDEX` |
| `argMin(axis, keepDimensions)`, `argMax(axis, keepDimensions)` | Caller choice | Explicitly supplies `FIRST_INDEX` |
| Complete three-argument forms | Caller choice | Exact non-null caller policy |

The convenience forms choose the smallest logical index among equal extrema by passing
`ArgExtremaTiePolicy.FIRST_INDEX` directly to the shared construction boundary. This is a
public-overload convenience, not an implicit default in `ArgExtremaAttrs`: the semantic attribute
record requires an explicit non-null policy. A complete overload can pass `LAST_INDEX`, which
requests the largest logical index among equal candidates. Logical indices are coordinates along
the normalized selected axis, independent of storage offset, stride, layout, or traversal.

Every form accepts `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, or `INT64` input and rejects BOOL;
numeric inputs are neither promoted nor converted. The positive or negative caller axis is
normalized exactly once against the input `Shape`. A false `keepDimensions` removes the selected
axis; true replaces only that axis with a new static extent of one. Removing the sole axis of a
rank-one tensor produces the canonical scalar Shape. Every unaffected static or dynamic
`Dimension` object is retained by exact reference.

Every result is a fresh `DataType.INT64` Tensor with false gradient eligibility, unresolved
layout, no label, and no host storage. Provenance records the exact
`AggregateReductionKind.ARG_MIN` or `ARG_MAX`, one
`ArgExtremaAttrs(normalizedAxis, keepDimensions, tiePolicy)`, exactly the receiver as its sole
ordered input, and output index zero. Repeated or nested calls are not simplified or canonicalized.

Integral candidates use signed order. Floating candidates use one shared total selection policy:
NaN is preferred to every non-NaN for both minimum and maximum; multiple NaNs are ties; negative
zero orders below positive zero; infinities retain their usual order; and otherwise equal numeric
values are ties. The explicit policy chooses the first or last logical index after this ordering.
This fixes semantic results without choosing an algorithm or traversal.

A statically known zero selected extent is rejected because it has no valid logical index, with
`IllegalArgumentException` and message `arg-extrema reduction axis must be non-empty, but axis
<normalizedAxis> has static extent 0`. A zero on an unselected axis is valid and may make the
result empty. A dynamic or expression selected extent is accepted structurally and must be
positive when later proven or bound; no sentinel index represents an empty domain. Construction
does not read storage, compare values, select an index, create gradients, capture a graph, lower an
operation, report backend support, or execute.

Failure behavior is deterministic: a null explicit policy fails before input-type or axis
validation; BOOL input fails before axis validation; and an axis outside
`[-rank, rank - 1]` fails with `IndexOutOfBoundsException`, including every axis for a scalar
input. Axis validation precedes static-zero inspection. These local failures occur before result
identity allocation.

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

### Slice expressions

`Tensor.slice(long[] starts, long[] ends, int[] axes, long[] steps)` constructs a fresh general
directional half-open slice expression. The four arrays are parallel: entry `i` supplies inclusive
raw start `starts[i]`, directional exclusive raw end `ends[i]`, input axis `axes[i]`, and signed
non-zero increment `steps[i]`. Positive steps select `start, start + step, ...` while the coordinate
is below the end; negative steps select while it is above the end. Array lengths must match, and
entry order is preserved. After checking the references and lengths, construction clones every
array before inspecting elements; it neither mutates nor retains caller-owned arrays.

Each negative axis adds the input rank once and must then be in range. Axes must be distinct after
normalization. Every selected dimension must be a `StaticDimension`; an unselected static or
dynamic `Dimension` is retained by exact reference. For extent `D`, a negative bound adds `D`
exactly once. Positive-step start and end then clamp to `[0, D]`. For a negative step and non-zero
`D`, the start clamps to `[0, D - 1]` and the exclusive end to `[-1, D - 1]`. Raw `-1` is still a
relative coordinate and becomes `D - 1`; it is not an end sentinel. Selecting through coordinate
zero therefore needs an end that normalizes below zero, such as raw `-D - 1` when representable.

Checked formulas calculate selected lengths without negating `Long.MIN_VALUE`:

```text
step > 0 and start < end: length = 1 + (end - 1 - start) / step
step < 0 and start > end: length = 1 - ((start - 1 - end) / step)
otherwise:                length = 0
```

Normalized `SliceAttrs` stores `start`, `length`, normalized axis, and unchanged signed step—not a
normalized end. Empty entries use canonical start zero and length zero. A negative-step selected
zero extent is canonical empty without bound arithmetic. Result rank is unchanged, each selected
axis receives a new static length, and every unselected Dimension reference remains exact.

Four empty arrays are valid and create a fresh explicit identity slice, including for a scalar.
An empty selection is also valid and produces a zero extent. Empty results deliberately have
unresolved layout: they reference no storage element, so recording one-past-end offset and stride
facts would be arbitrary and could overflow without adding observable geometry.

When input layout is unresolved, result layout remains unresolved. When input layout is resolved
and the result is non-empty and every step is positive, every resolved input kind—dense
contiguous, dense with offset, strided, or broadcast zero-stride—produces one new view-marked
logical `LayoutDescriptor`. Starting from the input offset and original strides, entry `i`
performs checked calculations:

```text
resultOffset       += normalizedStart[i] * originalInputStride[axis[i]]
resultStride[axis]  = originalInputStride[axis[i]] * step[i]
```

`LayoutDescriptor` reclassifies the resulting geometry and calculates its referenced element
span. The view flag records logical view metadata only. Slice construction attaches no host
storage and does not promise a physical alias, zero-copy route, or executable view.
If any step is negative, the complete result layout is unresolved because the current descriptor
forbids negative strides. That boundary chooses neither copying nor a reverse kernel.

Every successful result preserves the exact input data type and gradient eligibility, has no
label or host storage, and records `SliceKind.SLICE`, one normalized `SliceAttrs`, one
identity-distinct producer with exact inputs `[input]`, one output descriptor, and provenance
output index zero. Identity, repeated, and nested requests remain explicit and consume one fresh
Tensor identifier each. Validation and checked arithmetic fail before allocation.

`Tensor.sliceAxis(int, long, long)` uses one entry with step `1L`.
`Tensor.sliceAxis(int, long, long, long)` uses one entry with the caller's signed step.
`Tensor.flip(int... axes)` clones explicit axes in caller order and creates one SLICE producer with
start `D - 1`, length `D`, and step `-1` per non-empty selected extent. A selected zero extent uses
start and length zero. Duplicate normalized axes fail; empty axes mean identity rather than all
axes and remain valid even for a scalar or unselected dynamic dimensions. There is no `FLIP` kind
or per-axis operation chain.

The official [ONNX Slice contract](https://onnx.ai/onnx/operators/onnx__Slice.html) provides
terminology evidence for signed steps and direction-dependent bound clamping. Synaptik retains
explicit arrays, requires selected static dimensions, and stores selected lengths without ONNX's
open-bound sentinel convention. The official [JAX flip reference](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.flip.html)
provides established axis-reversal terminology; Synaptik requires explicit axes and gives empty
varargs identity meaning.

#### Complete slice-expression example

##### Goal and inputs

Create positive, negative, mixed-direction, empty, zero-extent, and flip expressions, then inspect
their normalized metadata. The primary input has resolved canonical Shape `[2, 5]`; the example
observes model metadata only.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class SliceExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 5);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true));

        Tensor positive = input.sliceAxis(1, 1, 5, 2);
        Tensor reverse = input.sliceAxis(1, 4, -6, -1);
        Tensor explicitMinusOne = input.sliceAxis(1, 4, -1, -1);
        Tensor mixed = input.slice(
                new long[] {0, 4}, new long[] {2, -6},
                new int[] {0, 1}, new long[] {1, -1});
        Tensor flipped = input.flip(1, 0);

        Shape zeroShape = Shape.of(2, 0);
        Tensor zeroInput = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, zeroShape,
                Optional.of(LayoutDescriptor.contiguous(zeroShape)), true));
        Tensor zeroFlip = zeroInput.flip(1);

        System.out.println(positive.descriptor().shape() + " " + attrs(positive));
        System.out.println(reverse.descriptor().shape() + " " + attrs(reverse));
        System.out.println(explicitMinusOne.descriptor().shape()
                + " " + attrs(explicitMinusOne));
        System.out.println(mixed.descriptor().shape() + " " + attrs(mixed));
        System.out.println(flipped.descriptor().shape() + " " + attrs(flipped));
        System.out.println(zeroFlip.descriptor().shape() + " " + attrs(zeroFlip));
        System.out.println(positive.descriptor().layout().isPresent());
        System.out.println(reverse.descriptor().layout().isEmpty());
        System.out.println(flipped.provenance().orElseThrow().producer().outputCount());
        System.out.println(flipped.provenance().orElseThrow().outputIndex());
        System.out.println(flipped.provenance().orElseThrow().inputs().getFirst() == input);
    }

    private static SliceAttrs attrs(Tensor tensor) {
        return (SliceAttrs) tensor.provenance().orElseThrow().operation().attrs();
    }
}
```

##### Meaningful lines and intermediate results

- The positive request selects coordinates `[1, 3]`, so it stores start `1`, length `2`, and step
  `2` and derives Shape `[2, 2]` with resolved view geometry.
- The reverse request selects `[4, 3, 2, 1, 0]`. The explicit raw end `-1` instead normalizes to
  coordinate `4`, so `(4, -1, -1)` is empty and stores canonical start and length zero.
- The mixed request stores both axes in caller order and leaves layout unresolved because one step
  is negative. `flip(1, 0)` stores both negative-step sequences in that same order under one
  producer. The zero-extent flip stores canonical start and length zero.
- Every result retains `FLOAT32` and true gradient eligibility and remains unlabeled and
  storage-free.

##### Result and interpretation

The program prints:

```text
Shape[2, 2] SliceAttrs[starts=[1], lengths=[2], axes=[1], steps=[2]]
Shape[2, 5] SliceAttrs[starts=[4], lengths=[5], axes=[1], steps=[-1]]
Shape[2, 0] SliceAttrs[starts=[0], lengths=[0], axes=[1], steps=[-1]]
Shape[2, 5] SliceAttrs[starts=[0, 4], lengths=[2, 5], axes=[0, 1], steps=[1, -1]]
Shape[2, 5] SliceAttrs[starts=[4, 1], lengths=[5, 2], axes=[1, 0], steps=[-1, -1]]
Shape[2, 0] SliceAttrs[starts=[0], lengths=[0], axes=[1], steps=[-1]]
true
true
1
0
true
```

The output demonstrates directional normalization, canonical empty state, exact Shape metadata,
the positive-view/negative-unresolved layout boundary, and one-producer flip provenance. It does
not demonstrate values, physical storage aliasing, gradients, compiler capture or
canonicalization, materialization, backend lowering, ONNX mapping, or execution.

##### Failures and useful variations

- A null request array fails with `NullPointerException` naming that parameter. Unequal lengths
  fail before the arrays are cloned or the input descriptor is read.
- An axis outside rank after one normalization, the first repeated normalized axis, a zero step,
  or selection of a dynamic dimension fails with `IllegalArgumentException`.
- Every raw `long` bound is accepted, normalized once when negative, and clamped by direction.
  Starts empty in the selected direction produce canonical empty state, not a failure.
- `flip` rejects its first invalid or duplicate normalized axis and any selected dynamic
  Dimension. Empty flip axes select nothing and are valid identity semantics.
- Result element-count, offset, stride, layout-classification, or referenced-span overflow
  propagates as `ArithmeticException`. These local failures occur before Tensor identity
  allocation.
- Identifier exhaustion occurs only at final derived construction after normalized attributes,
  Shape, and optional layout have been created.

### Scalar select expressions

`Tensor.select(int axis, long index)` fixes one scalar coordinate on an existing source axis and
removes that axis from a fresh result. The operation is a single-result scalar selection, not
elementwise conditional `WHERE`, repeated-select unstack, half-open interval `SLICE`,
or a gather whose indices come from a Tensor.

The caller may address an input axis from the front with a non-negative value or from the end with
a negative value in `[-rank, rank - 1]`. Construction normalizes that axis exactly once. When the
selected Dimension is static, one negative index adds the selected extent once, and every
normalized index must be in `[0, extent)`. A selected zero extent therefore rejects every index.
When the selected Dimension is dynamic, a non-negative index is retained unchanged and its upper-
bound check is deferred; a negative index is rejected because no numeric extent exists for local
normalization.

The result Shape omits exactly the selected Dimension. Every unaffected static or dynamic
Dimension remains the exact input reference in its original order. Selecting the only axis of a
rank-one Tensor produces the canonical scalar Shape `[]`.

Unresolved input layout produces unresolved result layout. An empty result also stays unresolved
because it references no storage element and needs no arbitrary offset geometry. Otherwise,
resolved dense, offset, strided, or zero-stride input geometry produces one new view-marked
logical `LayoutDescriptor`: the selected stride is removed, and checked arithmetic advances the
element offset by `normalizedIndex * selectedStride`. This metadata describes a logical view; the
result has no host storage and does not promise a physical alias or zero-copy execution.

Every successful result preserves the exact input data type and gradient eligibility, has no
label or host storage, and records `SelectKind.SELECT`, one normalized
`SelectAttrs(normalizedAxis, normalizedIndex)`, and provenance `[input]`. Repeated, nested, and
same-coordinate requests remain distinct fresh expressions.

#### Complete scalar-select expression example

##### Goal and inputs

Select the final coordinate of axis one from a resolved `FLOAT32` Tensor with Shape `[2, 3, 4]`,
then inspect the normalized semantics and logical view geometry. The input has canonical strides
`[12, 4, 1]` and offset zero.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class ScalarSelectExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3, 4);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true));

        Tensor selected = input.select(-2, -1);
        SelectAttrs attrs = (SelectAttrs) selected.provenance().orElseThrow()
                .operation().attrs();
        LayoutDescriptor layout = selected.descriptor().layout().orElseThrow();

        System.out.println(selected.descriptor().shape());
        System.out.println(attrs);
        System.out.println(Arrays.toString(layout.strides()));
        System.out.println(layout.storageOffset());
        System.out.println(layout.isView());
        System.out.println(selected.provenance().orElseThrow().inputs().getFirst() == input);
        System.out.println(selected.descriptor().requiresGrad());
        System.out.println(selected.label().isEmpty() && selected.hostStorage().isEmpty());
    }
}
```

##### Meaningful lines and intermediate results

- Raw axis `-2` adds rank three once and becomes axis `1`. Its static extent is three, so raw index
  `-1` adds three once and becomes index `2`. The equivalent positive request is
  `select(1, 2)`.
- Removing axis one from Shape `[2, 3, 4]` leaves Shape `[2, 4]`. The input strides
  `[12, 4, 1]` lose selected stride `4`, producing `[12, 1]`.
- The checked result offset is `0 + 2 * 4 = 8`. The result retains `FLOAT32` and true gradient
  eligibility, records exact one-input provenance, and remains unlabeled and storage-free.

##### Result and interpretation

The program prints:

```text
Shape[2, 4]
SelectAttrs[axis=1, index=2]
[12, 1]
8
true
true
true
true
```

The output demonstrates raw axis/index normalization, axis removal, checked logical view geometry,
exact semantics, and fresh provenance. It does not demonstrate selected values, a physical storage
alias, a gradient rule, compiler capture, materialization, backend lowering, or execution.

##### Failures and useful variations

- Every axis is invalid for a scalar input. Other axes outside `[-rank, rank - 1]` fail through
  Shape normalization before result identity allocation.
- A static selected extent rejects an index below `-extent` or at least `extent` after one
  normalization. A dynamic selected extent accepts every non-negative `long` unchanged with its
  upper bound deferred, but rejects a negative value because it cannot normalize it locally.
- Selecting the only axis of Shape `[5]` produces scalar Shape `[]`. Selecting a dynamic axis
  removes it while preserving exact unaffected Dimension references, but current dynamic input
  descriptors cannot provide resolved numeric layout geometry.
- Result-element-count, offset, layout-classification, or referenced-span overflow propagates as
  `ArithmeticException` before Tensor identity allocation. Identifier exhaustion can occur only
  during final derived construction.

### Axis-gather expressions

The public axis-gather methods produce expressions with ordered logical inputs
`[data, indices]`. `data` is the receiver and supplies the result data type and gradient
eligibility. Both require exact `INT32` or `INT64` indices; no floating or BOOL index tensor is
accepted or implicitly cast. `gather` and `gatherElements` normalize one positive or negative axis
exactly once against the data Shape after their null and index-type checks; `embedding` fixes that
axis at zero after validating its rank-two floating receiver.

The methods differ in how they relate the two input Shapes to the result:

| Public method | Semantic kind | Shape rule |
|---|---|---|
| `gather(indices, axis)` | `GATHER` | Replace the selected data Dimension with every indices Dimension in order. |
| `embedding(indices)` | `GATHER` at axis zero | Require rank-two floating weights and append their exact axis-one embedding Dimension to the complete indices Shape. |
| `gatherElements(indices, axis)` | `GATHER_ELEMENTS` | Require equal ranks and equal Dimensions on every non-selected axis, then retain the exact indices Shape as the result. |

Structural equality is conservative for dynamic Dimensions. The same dynamic symbol passes where
equality is required; different symbols fail rather than creating a new symbolic constraint.
`gather` needs no compatibility check between the inserted indices Shape and the omitted data
Dimension, and scalar indices remove that axis without inserting another. `gatherElements`
permits the selected indices extent to differ from the data extent.

Every successful request returns a fresh unlabeled, storage-free result Tensor with unresolved
layout, even when both provenance inputs have resolved geometry. Its provenance contains the exact
semantic kind, one `IndexAxisAttrs` with the normalized axis, and exact input references in
`[data, indices]` order.
Expression construction never interprets, normalizes, clamps, or bounds-checks index values. It
defines no gradient or repeated-index rule, compiler capture or canonicalization, materialization,
backend lowering, or execution behavior.

#### Embedding convenience

`weights.embedding(indices)` gives the common lookup-table use of Gather a direct public spelling.
The receiver is a rank-two floating table shaped `[vocabulary, embeddingSize]`; indices may have
any rank but must use exact `INT32` or `INT64`. The result Shape is:

```text
indices.shape + [weights.shape[1]]
```

Every indices Dimension is retained by exact reference and in its original order, followed by the
exact weight axis-one Dimension. Scalar indices therefore produce `[embeddingSize]`. Result type
and gradient eligibility come only from weights. Layout remains unresolved, and the result has no
label or storage.

##### Complete embedding metadata example

The goal is to construct lookup metadata for weights shaped `[10, 4]` and indices shaped `[2, 3]`
and inspect the ordinary Gather occurrence. Both inputs are storage-free because construction
uses descriptors, not element values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class EmbeddingExpressionExample {
    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }

    public static void main(String[] args) {
        Tensor weights = tensor(DataType.FLOAT32, Shape.of(10, 4), true);
        Tensor indices = tensor(DataType.INT64, Shape.of(2, 3), false);

        Tensor embedded = weights.embedding(indices);
        var provenance = embedded.provenance().orElseThrow();

        System.out.println(embedded.descriptor().shape());
        System.out.println(provenance.operation().kind());
        System.out.println(provenance.operation().attrs());
        System.out.println(provenance.inputs().get(0) == weights);
        System.out.println(provenance.inputs().get(1) == indices);
        System.out.println(provenance.outputIndex());
        System.out.println(embedded.descriptor().requiresGrad());
        System.out.println(embedded.descriptor().layout().isEmpty()
                && embedded.label().isEmpty()
                && embedded.hostStorage().isEmpty());
    }
}
```

The indices axes `2` and `3` are retained first, and the weight embedding extent `4` is appended.
The program prints:

```text
Shape[2, 3, 4]
GATHER
IndexAxisAttrs[axis=0]
true
true
0
true
true
```

This proves the derived Shape, direct axis-zero Gather semantics, ordered `[weights, indices]`
provenance, single output at index zero, weight-only gradient eligibility, and storage-free result
metadata. Each valid call creates one producer and one fresh Tensor ID. It does not read the six
index values, select weight rows, construct a gradient, compile a graph, or execute work.

Negative and out-of-range stored index values remain accepted at construction because model code
does not inspect values. They are invalid for later ordinary Gather execution and must not wrap,
clamp, select padding, or select a default row. Compiler constant analysis may reject values it can
prove invalid; after dynamic extents are bound, preparation or prepared execution must enforce the
ordinary Gather bounds contract safely. This convenience adds no `EMBEDDING` kind and no
padding-index, sparse-gradient, maximum-norm, or frequency-scaling option.

#### Complete axis-gather expression example

##### Goal and inputs

Compare both Shape relationships for `FLOAT32` data with Shape `[2, 3, 4]`. The example uses
storage-free index tensors because current expression construction needs only their immutable
descriptors.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class AxisGatherExpressionExample {
    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }

    public static void main(String[] args) {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor insertedIndices = tensor(DataType.INT64, Shape.of(5, 6), false);
        Tensor alignedIndices = tensor(DataType.INT32, Shape.of(2, 7, 4), false);

        Tensor gathered = data.gather(insertedIndices, -2);
        Tensor elements = data.gatherElements(alignedIndices, 1);

        IndexAxisAttrs attrs = (IndexAxisAttrs) gathered.provenance().orElseThrow()
                .operation().attrs();
        System.out.println(gathered.descriptor().shape());
        System.out.println(gathered.provenance().orElseThrow().operation().kind());
        System.out.println(elements.descriptor().shape());
        System.out.println(elements.provenance().orElseThrow().operation().kind());
        System.out.println(attrs);
        System.out.println(gathered.provenance().orElseThrow().inputs().get(0) == data);
        System.out.println(gathered.provenance().orElseThrow().inputs().get(1) == insertedIndices);
        System.out.println(gathered.descriptor().requiresGrad());
        System.out.println(gathered.descriptor().layout().isEmpty()
                && gathered.label().isEmpty()
                && gathered.hostStorage().isEmpty());
    }
}
```

##### Meaningful lines and intermediate results

- Raw axis `-2` normalizes once against rank three to axis `1`.
- `gather` inserts complete Shape `[5, 6]` where extent `3` was, producing `[2, 5, 6, 4]`.
- `gatherElements` verifies the aligned extents `2` and `4`, ignores the selected extents `3` and
  `7`, and retains exact indices Shape `[2, 7, 4]`.
- All results retain `FLOAT32` and true gradient eligibility from `data`; the index tensors do not
  contribute either fact.

##### Result and interpretation

The program prints:

```text
Shape[2, 5, 6, 4]
GATHER
Shape[2, 7, 4]
GATHER_ELEMENTS
IndexAxisAttrs[axis=1]
true
true
true
true
```

The output demonstrates axis normalization, both Shape rules, metadata retention, unresolved
layout, and ordered provenance. It does not demonstrate
gathered values, valid index-value bounds, gradient behavior, compiler capture, backend lowering,
or execution.

##### Failures and useful variations

- Null data and indices fail in that order. An unsupported indices data type fails before axis
  normalization. An invalid axis then fails through `Shape.normalizeAxis`.
- `gatherElements` checks rank first, then non-axis Dimensions in increasing axis order and reports
  the first mismatch. Its selected-axis extents may differ.
- Scalar indices passed to `gather` remove the selected data axis without inserting another one.
  Rank-one `gather` with scalar indices produces scalar Shape `[]`.
- For both signatures, local validation completes before result identity
  allocation. Identifier exhaustion can occur only during final derived construction.

### One-hot encoding expressions

`indices.oneHot(depth)` creates metadata for one dense trailing-axis indicator encoding. The
receiver must have exact INT32 or INT64 type, and `depth` must be a positive `long`. For receiver
Shape `[d0, ..., dn]`, the result Shape is:

```text
[exact d0 reference, ..., exact dn reference, new StaticDimension(depth)]
```

Every existing Dimension object is retained by reference and in order. The appended Dimension is
fresh. A scalar receiver therefore produces Shape `[depth]`; a receiver containing a zero extent
remains valid and has zero logical rows. `Long.MAX_VALUE` depth is structurally accepted because
construction does not calculate a result element count or materialize storage.

For an input coordinate `p` with eventual index value `i`, the exact logical meaning is:

```text
result[p..., j] = (i == j), for 0 <= j < depth
```

The result has exact BOOL type, `requiresGrad=false`, unresolved layout, no label, and no host
storage. Each successful call creates one `ONE_HOT` operation with `OneHotAttrs(depth)`, one fresh
single-output producer retaining exact input `[indices]`, provenance output index zero, and one
fresh Tensor identity.

#### Conceptual vector and scalar examples

The goal is to connect index values to the requested logical result while keeping the current
metadata-only boundary explicit. For conceptual INT64 values `[2, 0, 1]` with Shape `[3]` and
depth `3`, the requested values are:

```text
input:  [2, 0, 1]
result: [[false, false, true],
         [true,  false, false],
         [false, true,  false]]
```

Each input position becomes one row on the new trailing class axis. Index `2` selects trailing
position `2`, index `0` selects position `0`, and index `1` selects position `1`. The result Shape
is `[3, 3]`. A conceptual scalar INT32 index `2` with depth `4` instead requests rank-one Shape
`[4]` and values `[false, false, true, false]`.

These value arrays are conceptual expected results, not output produced by current code.
`oneHot` inspects only receiver metadata, so it does not read the example values, allocate the
BOOL result, compile an operation, or execute encoding. A caller can inspect the current
construction contract with storage-free metadata:

```java
Tensor indices = TensorFactory.create(new TensorDescriptor(
        DataType.INT64, Shape.of(3), Optional.empty(), false));
Tensor encoded = indices.oneHot(3);

System.out.println(encoded.descriptor().shape());
System.out.println(encoded.descriptor().dataType());
System.out.println(encoded.provenance().orElseThrow().operation());
```

The observable metadata is Shape `[3, 3]`, BOOL, and an operation whose kind is `ONE_HOT` and
whose attributes are `OneHotAttrs[depth=3]`. This proves Shape, type, and semantic provenance; it
does not prove or calculate the conceptual BOOL values.

#### Failures and execution boundary

- A receiver whose type is not INT32 or INT64 fails before depth validation with
  `oneHot indices data type must be INT32 or INT64: <type>`.
- Depth zero or a negative depth fails with `depth must be positive: <depth>`. Null helper input,
  type failure, and depth failure occur before Tensor identity allocation.
- Valid eventual execution requires `0 <= i < depth` for every index value. Negative and
  out-of-range values are invalid; they do not wrap, clamp, select a default, or produce an
  all-false row. Model construction deliberately accepts stored invalid values because it never
  reads storage. A later compiler may reject provably invalid constants, while future preparation
  or prepared execution must safely enforce the bound for dynamically supplied values.
- There is no configurable axis, Tensor-valued depth, on/off value, output type, ignore index,
  sparse form, gradient rule, compiler support, backend lowering, or runtime implementation.

### Gather-ND expressions

The two public Gather-ND methods consume ordered logical inputs `[data, indices]`. The receiver is
`data`; `indices` must use exact `INT32` or `INT64` elements and have rank at least one. The
one-argument method is exactly the zero-batch form of `gatherNd(indices, batchDimensions)`.

The explicit non-negative batch count `B` must be smaller than both input ranks. Data and indices
Dimensions on axes `[0, B)` must be structurally equal: equal static sizes and equal dynamic
symbols pass, while broadcasting and new symbolic constraints are not introduced. The final
indices Dimension must have a statically known positive extent `K`, the tuple depth, no greater
than `dataRank - B`. Each tuple indexes data axes `[B, B + K)`.

For data rank `R` and indices rank `Q`, expression construction creates the result Shape once as:

```text
indices.shape[0:Q-1] + data.shape[B+K:R]
```

The result retains the exact Dimension references from the indices prefix and untouched data
suffix. If both parts are empty, `Shape.ofDimensions` returns the canonical rank-zero scalar
Shape. Every valid request creates a fresh, unlabeled, storage-free Tensor with the data Tensor's
exact type and gradient eligibility, unresolved layout, `GatherNdKind.GATHER_ND`, one exact
`GatherNdAttrs(B)`, and provenance containing exact references in `[data, indices]` order.

#### Complete Gather-ND expression example

##### Goal and inputs

Construct zero-batch, batched, scalar-result, and dynamic-Dimension Gather-ND expressions. The
index tensors are storage-free because metadata construction needs their descriptors, not their
values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class GatherNdExpressionExample {
    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }

    public static void main(String[] args) {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor zeroBatchIndices = tensor(DataType.INT32, Shape.of(5, 2), false);
        Tensor batchedIndices = tensor(DataType.INT64, Shape.of(2, 5, 1), false);
        Tensor zeroBatch = data.gatherNd(zeroBatchIndices);
        Tensor batched = data.gatherNd(batchedIndices, 1);

        Tensor scalar = tensor(DataType.BOOL, Shape.of(2, 3), false)
                .gatherNd(tensor(DataType.INT32, Shape.of(2), false));

        DynamicDimension batch = new DynamicDimension("N");
        DynamicDimension queries = new DynamicDimension("M");
        StaticDimension suffix = new StaticDimension(4);
        Tensor dynamicData = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(batch, new StaticDimension(3), suffix), true);
        Tensor dynamicIndices = tensor(DataType.INT64,
                Shape.ofDimensions(new DynamicDimension("N"), queries,
                        new StaticDimension(1)), false);
        Tensor dynamic = dynamicData.gatherNd(dynamicIndices, 1);

        GatherNdAttrs attrs = (GatherNdAttrs) batched.provenance().orElseThrow()
                .operation().attrs();
        System.out.println(zeroBatch.descriptor().shape());
        System.out.println(batched.descriptor().shape());
        System.out.println(scalar.descriptor().shape() == Shape.scalar());
        System.out.println(dynamic.descriptor().shape());
        System.out.println(attrs);
        System.out.println(dynamic.descriptor().shape().dimensions().get(1) == queries);
        System.out.println(dynamic.descriptor().shape().dimensions().get(2) == suffix);
        System.out.println(batched.provenance().orElseThrow().inputs().get(0) == data);
        System.out.println(batched.provenance().orElseThrow().inputs().get(1)
                == batchedIndices);
        System.out.println(batched.descriptor().requiresGrad()
                && batched.descriptor().layout().isEmpty()
                && batched.label().isEmpty()
                && batched.hostStorage().isEmpty());
    }
}
```

##### Meaningful lines and intermediate results

- Zero-batch indices `[5, 2]` have tuple depth `2`. They retain prefix `[5]`, index data axes `0`
  and `1`, and leave data suffix `[4]`, producing `[5, 4]`.
- Batched indices `[2, 5, 1]` share leading Dimension `2`, retain indices prefix `[2, 5]`, index
  data axis `1`, and leave suffix `[4]`, producing `[2, 5, 4]`.
- Indices Shape `[2]` over data Shape `[2, 3]` has no retained indices prefix or data suffix, so
  the result is the canonical scalar `Shape.scalar()`.
- The dynamic case accepts independently constructed but structurally equal `N` batch Dimensions.
  Its result `[N, M, 4]` retains the exact `M` reference from indices and exact suffix Dimension
  reference from data.

##### Result and interpretation

The program prints:

```text
Shape[5, 4]
Shape[2, 5, 4]
true
Shape[N, M, 4]
GatherNdAttrs[batchDimensions=1]
true
true
true
true
true
```

The output demonstrates both overloads, all four Shape cases, exact retained metadata and
Dimension identities, normalized attributes, and ordered provenance. It does not demonstrate
index-value validity, gathered values, gradient behavior, compiler capture, materialization,
backend lowering, or execution.

##### Failures and useful variations

- Validation checks null data and indices, index type, indices rank, non-negative batch count,
  indices-rank fit, data-rank fit, batch-prefix equality, and tuple depth in that order. Every
  local failure occurs before result identifier allocation.
- A dynamic final indices Dimension fails because tuple depth controls which data axes remain in
  the result. Static depth zero or a depth greater than `dataRank - B` also fails.
- Different dynamic batch symbols fail structural equality; no broadcast or equality constraint
  is recorded.
- Construction never reads, normalizes, clamps, or bounds-checks an index value. Negative and
  out-of-range coordinate policy remains outside this model metadata boundary.
- Checked result-rank overflow propagates as `ArithmeticException`; identifier exhaustion can
  occur only during final derived Tensor construction.

### Scatter Elements expressions

The two public `scatterElements` overloads construct a functional same-rank update from ordered
logical inputs `[data, indices, updates]`. The receiver is unchanged base `data`; `indices`
selects one target coordinate along the normalized axis, and `updates` supplies the corresponding
value. The short overload selects `ScatterReduction.NONE`; the complete overload retains the
caller's exact non-null reduction.

Indices must use exact `INT32` or `INT64`, and updates must use the data tensor's exact type.
Indices and updates must have equal rank and structurally equal Shapes, that rank must equal the
data rank, and every non-selected indices Dimension must equal the corresponding data Dimension.
The selected extent may differ. For data Shape `[2, 3, 4]`, axis `1`, and indices/updates Shape
`[2, 5, 4]`, the result has exact data Shape `[2, 3, 4]`. At update coordinate `[0, 4, 2]`,
the corresponding index conceptually selects target `[0, index, 2]`.

`NONE` permits every current data type and requires unique targets. `ADD`, `MUL`, `MAX`,
and `MIN` permit floating and integral data and reject `BOOL`. Duplicate detection needs index
values and remains outside this metadata-only boundary.

Every valid call returns a fresh, unlabeled, storage-free Tensor with exact data Shape/type,
data/update gradient-eligibility OR, unresolved layout, `AxisScatterKind.SCATTER_ELEMENTS`,
`ScatterElementsAttrs(normalizedAxis, reduction)`, and exact ordered provenance. Construction
reads no values, checks no index bound or duplicate target, applies no write or reduction, mutates
no input, and defines no gradient, compiler, materialization, backend, or execution behavior.

#### Complete Scatter Elements expression example

##### Goal and inputs

Construct replacement and maximum-reduction expressions for `FLOAT32` data Shape `[2, 3, 4]`
using aligned indices and updates Shape `[2, 5, 4]`.

```java
Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
Tensor indices = tensor(DataType.INT32, Shape.of(2, 5, 4), false);
Tensor updates = tensor(DataType.FLOAT32, Shape.of(2, 5, 4), false);

Tensor replacement = data.scatterElements(indices, updates, -2);
Tensor maximum = data.scatterElements(indices, updates, 1, ScatterReduction.MAX);
```

##### Result and interpretation

Both results have Shape `[2, 3, 4]`. The negative axis normalizes to `1`; replacement records
`ScatterElementsAttrs[axis=1, reduction=NONE]`, while maximum records `MAX`. This proves
structural expression construction and reduction retention, not indexed writes or execution.

##### Failures and useful variations

Null data, indices, updates, and explicit reduction fail in that order. Index type fails before
data/update type equality, reduction eligibility, axis normalization, and Shape checks. Shape
validation checks indices rank, updates rank, every updates/indices Dimension, then every
non-selected indices/data Dimension, all in increasing-axis order. Local failures occur before
result identity allocation.
### Scatter-ND expressions

The three public Scatter-ND overloads construct functional tuple-index updates from ordered
logical inputs `[data, indices, updates]`. `data` is the receiver and unchanged base value.
`indices` supplies coordinate tuples, and `updates` supplies one value or suffix slice for each
tuple position. A target is the result coordinate or suffix slice addressed by one tuple; a
duplicate target occurs when multiple tuples address that same target.

The overloads delegate to one validation path:

| Public method | Reduction | Shared batch count |
|---|---|---|
| `scatterNd(indices, updates)` | `NONE` | `0` |
| `scatterNd(indices, updates, reduction)` | Exact supplied reduction | `0` |
| `scatterNd(indices, updates, reduction, batchDimensions)` | Exact supplied reduction | Exact supplied non-negative count |

All paths require exact `INT32` or `INT64` indices and exact matching data/update types.
`ScatterReduction.NONE` accepts every current data type. `ADD`, `MUL`, `MAX`, and `MIN` accept
floating or integral data and reject `BOOL`; no input is converted or promoted.

For data rank `R`, indices rank `Q`, shared-batch count `B`, and tuple depth `K` from the final
indices Dimension, construction requires `0 <= B < Q`, `B < R`, structurally equal data and
indices Dimensions on axes `[0, B)`, and `1 <= K <= R - B`. Tuple depth must be statically known.
The exact updates Shape is:

```text
indices.shape[0:Q-1] + data.shape[B+K:R]
```

The result retains the exact data Shape and type, combines data/update gradient eligibility by
logical OR, and has unresolved layout, no label, and no storage. Provenance records
`ScatterNdKind.SCATTER_ND`, one exact `ScatterNdAttrs(B, reduction)`, and the exact ordered
references `[data, indices, updates]`. Every valid request is fresh, including repeated requests
with identical operands and attributes.

#### Complete Scatter-ND expression example

##### Goal and inputs

Construct zero-batch replacement, zero-batch addition, explicit batched maximum, and scalar-update
Scatter-ND expressions. The inputs are storage-free because construction needs only immutable
descriptor metadata; it does not read any index or update value.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class ScatterNdExpressionExample {
    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }

    public static void main(String[] args) {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false);
        Tensor zeroBatchIndices = tensor(DataType.INT32, Shape.of(5, 2), false);
        Tensor zeroBatchUpdates = tensor(DataType.FLOAT32, Shape.of(5, 4), true);
        Tensor replacement = data.scatterNd(zeroBatchIndices, zeroBatchUpdates);
        Tensor addition = data.scatterNd(
                zeroBatchIndices, zeroBatchUpdates, ScatterReduction.ADD);

        Tensor batchedIndices = tensor(DataType.INT64, Shape.of(2, 5, 1), false);
        Tensor batchedUpdates = tensor(DataType.FLOAT32, Shape.of(2, 5, 4), false);
        Tensor batched = data.scatterNd(
                batchedIndices, batchedUpdates, ScatterReduction.MAX, 1);

        Tensor boolData = tensor(DataType.BOOL, Shape.of(2, 3), false);
        Tensor scalarUpdates = tensor(DataType.BOOL, Shape.scalar(), false);
        Tensor scalar = boolData.scatterNd(
                tensor(DataType.INT32, Shape.of(2), false), scalarUpdates);

        TensorProvenance origin = replacement.provenance().orElseThrow();
        ScatterNdAttrs defaultAttrs = (ScatterNdAttrs) origin.operation().attrs();
        ScatterNdAttrs additionAttrs = (ScatterNdAttrs) addition
                .provenance().orElseThrow().operation().attrs();
        ScatterNdAttrs batchedAttrs = (ScatterNdAttrs) batched
                .provenance().orElseThrow().operation().attrs();

        System.out.println(zeroBatchUpdates.descriptor().shape());
        System.out.println(replacement.descriptor().shape());
        System.out.println(batchedUpdates.descriptor().shape());
        System.out.println(batched.descriptor().shape());
        System.out.println(scalarUpdates.descriptor().shape() == Shape.scalar());
        System.out.println(scalar.descriptor().shape());
        System.out.println(defaultAttrs);
        System.out.println(additionAttrs);
        System.out.println(batchedAttrs);
        System.out.println(origin.inputs().get(0) == data
                && origin.inputs().get(1) == zeroBatchIndices
                && origin.inputs().get(2) == zeroBatchUpdates);
        System.out.println(replacement.descriptor().dataType() == DataType.FLOAT32
                && replacement.descriptor().requiresGrad()
                && replacement.descriptor().layout().isEmpty()
                && replacement.label().isEmpty()
                && replacement.hostStorage().isEmpty());
        System.out.println(replacement != addition
                && !replacement.id().equals(addition.id()));
    }
}
```

##### Meaningful lines and intermediate results

- Zero-batch indices `[5, 2]` have tuple depth `2`. Their prefix `[5]` combines with the untouched
  data suffix `[4]`, so updates must be `[5, 4]`; the result remains data-shaped `[2, 3, 4]`.
- Batched indices `[2, 5, 1]` share leading Dimension `2`, use tuple depth `1` to address data axis
  `1`, and retain suffix `[4]`. Updates are `[2, 5, 4]`, and `ScatterNdAttrs` retains batch count
  `1` and `MAX`.
- Indices `[2]` over BOOL data `[2, 3]` have tuple depth `2` and leave no prefix or suffix, so the
  required updates Shape is the canonical scalar `Shape.scalar()`. The result remains `[2, 3]`.
- The short overload records `NONE` and batch count zero. The reduction overload records `ADD`
  and batch count zero. The true update eligibility on `zeroBatchUpdates` makes the replacement
  result eligible; indices never contribute eligibility.

##### Result and interpretation

The program prints:

```text
Shape[5, 4]
Shape[2, 3, 4]
Shape[2, 5, 4]
Shape[2, 3, 4]
true
Shape[2, 3]
ScatterNdAttrs[batchDimensions=0, reduction=NONE]
ScatterNdAttrs[batchDimensions=0, reduction=ADD]
ScatterNdAttrs[batchDimensions=1, reduction=MAX]
true
true
true
```

The output proves overload defaults, both Shape formulas, canonical scalar updates, exact
attributes, ordered provenance, data/update eligibility OR, unresolved storage-free result
metadata, and fresh identity. It does not prove that an index is in bounds, detect duplicate
targets, apply replacement or reduction, calculate values, construct gradients, capture a graph,
select a backend, or execute work.

##### Failures and useful variations

- Validation checks null data, indices, updates, and reduction; exact index type; exact
  data/update type; reduction eligibility; indices rank; non-negative batch count; indices-rank
  fit; data-rank fit; batch-prefix equality; tuple depth; and updates Shape in that order. Every
  local failure occurs before result identity allocation.
- A dynamic final indices Dimension fails because tuple depth determines which data axes are
  indexed. Static depth zero or greater than `dataRank - batchDimensions` also fails.
- Structurally equal dynamic batch symbols pass. Different symbols fail; construction creates no
  broadcast or compiler-style equality constraint.
- `BOOL` accepts `NONE` but rejects every arithmetic reduction before Shape validation. Floating
  and integral data accept all five reductions.
- Construction never reads, normalizes, clamps, or bounds-checks an index value. It does not
  detect the duplicate targets that make `NONE` invalid. Identifier exhaustion can occur only at
  final derived Tensor construction.

### Pad and tile expressions

`Tensor.pad(long[] before, long[] after, ScalarValue constantValue)` constructs a fresh constant-
pad expression. The two arrays must contain exactly one non-negative width per input axis. Construction
checks both references and ranks, clones both arrays, and then validates and derives metadata from
the private copies; it never retains or mutates caller-owned arrays. At axis `i`, a static input
extent produces this checked result extent:

```text
inputExtent + before[i] + after[i]
```

For example, Shape `[2, 3]`, before widths `[1, 0]`, and after widths `[2, 4]` produce Shape
`[5, 7]`. Static zero extents and empty arrays for a scalar are valid. Every Dimension category
uses the same canonical symbolic arithmetic. Zero widths preserve the exact input Dimension
reference, while non-zero widths are retained as an exact formula.

The supplied padding value must exactly match the input data type and is retained by exact
reference without conversion. All six current data types are accepted, including exact INT64
values above `2^53`, raw BFLOAT16 patterns, and canonical BOOL. The retained `double` overload
constructs an exact FLOAT64 value, so it succeeds only for a FLOAT64 receiver. Neither overload
decides whether a future backend can represent or execute the request.

`Tensor.tile(long... repeats)` constructs a fresh complete-pattern tiling expression. Its array
must contain exactly one strictly positive repeat count per input axis, is cloned before element
inspection, and is neither retained nor mutated. At axis `i`, a static input extent produces the
checked product `inputExtent * repeats[i]`. Shape `[2, 3]` with repeats `[2, 4]` therefore produces
Shape `[4, 12]`. Tiling repeats the complete logical pattern, not each scalar into one consecutive
run. A static zero extent and an empty scalar request are valid. A dynamic dimension is retained
by exact reference when its repeat is one; any other positive repeat is retained as an exact
canonical symbolic product.

#### Symbolic Shape examples

##### Goal and inputs

Show how pad and tile preserve exact Shape relationships before a concrete value for named extent
`N` is known. The input Shape is `[N]`.

```text
pad before [2], after [3]:  N -> N + 5
tile repeats [4]:           N -> 4 * N
```

##### Result and interpretation

Padding applies the before width and then the after width through checked symbolic addition;
tiling applies the positive repeat through checked symbolic multiplication. The resulting Shapes
retain `[N + 5]` and `[4 * N]`. These formulas describe logical extents only: construction does
not bind `N`, calculate a concrete element count, derive physical layout, materialize values, or
execute either operation. Checked static values, symbolic coefficients, and symbolic offsets fail
with `ArithmeticException` before result identity allocation if they overflow `long`.

Both methods preserve the input's exact data type and gradient-eligibility value. Every successful
call returns a distinct unlabeled, storage-free Tensor with unresolved layout, even for an
identity-like request and even when the input has resolved canonical, offset, strided, or
broadcast geometry. Padding and tiling require future output materialization; unlike reshape,
expand, permutation, rank editing, or non-empty slice, they are not ordinary input-view geometry.
The results respectively record `PadKind.PAD` with one normalized `PadAttrs`, or `TileKind.TILE`
with one normalized `TileAttrs`, and exact provenance `[input]`.

#### Complete pad-and-tile expression example

##### Goal and inputs

Construct both expressions from one resolved Tensor and inspect their checked Shapes, immutable
attributes, operation identity, and deliberately unresolved storage-free results.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class PadTileExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true));

        long[] before = {1, 0};
        long[] after = {2, 4};
        long[] repeats = {2, 4};
        Tensor padded = input.pad(before, after, ScalarValue.float32(-1.0f));
        Tensor tiled = input.tile(repeats);
        before[0] = 99;
        repeats[0] = 99;

        PadAttrs padAttrs = (PadAttrs) padded.provenance().orElseThrow()
                .operation().attrs();
        TileAttrs tileAttrs = (TileAttrs) tiled.provenance().orElseThrow()
                .operation().attrs();

        System.out.println(padded.descriptor().shape());
        System.out.println(padAttrs.before());
        System.out.println(padAttrs.after());
        System.out.println(padAttrs.constantValue());
        System.out.println(tiled.descriptor().shape());
        System.out.println(tileAttrs.repeats());
        System.out.println(padded.provenance().orElseThrow().operation().kind());
        System.out.println(tiled.provenance().orElseThrow().operation().kind());
        System.out.println(padded.provenance().orElseThrow().inputs().getFirst() == input
                && tiled.provenance().orElseThrow().inputs().getFirst() == input);
        System.out.println(padded.descriptor().layout().isEmpty()
                && tiled.descriptor().layout().isEmpty());
        System.out.println(padded.hostStorage().isEmpty()
                && tiled.hostStorage().isEmpty());
    }
}
```

##### Meaningful lines and intermediate results

- Checked addition derives padded Shape `[5, 7]`; checked multiplication derives tiled Shape
  `[4, 12]`.
- Mutating the input arrays after construction cannot change the normalized attributes, which
  remain `[1, 0]`, `[2, 4]`, and `[2, 4]`.
- The padding constant is an exact FLOAT32 `ScalarValue`, retained without a binary64 intermediate.
- Both expressions retain `FLOAT32` and true gradient eligibility, record their distinct semantic
  kinds, and preserve the exact input Tensor in one-input provenance.
- The resolved input geometry does not propagate. Both fresh results remain unresolved and have
  no host storage because these calls describe materializing operations without performing them.

##### Result and interpretation

The program prints:

```text
Shape[5, 7]
[1, 0]
[2, 4]
ScalarValue[dataType=FLOAT32, bits=0xBF800000]
Shape[4, 12]
[2, 4]
PAD
TILE
true
true
true
```

This output demonstrates rank-coupled request ownership, checked Shape derivation, exact semantic
attributes, one-input provenance, and unresolved storage-free results. It does not demonstrate
padding or repeated values, padding-constant conversion, gradient rules, graph capture,
materialization, compiler canonicalization, backend or ONNX lowering, or execution.

##### Failures and useful variations

- A null array or typed padding value fails with `NullPointerException` naming that parameter. A
  length different from input rank, a negative padding width, a mismatched constant type, or a
  non-positive repeat fails with
  `IllegalArgumentException` before result identity allocation.
- Static extent, symbolic coefficient, or symbolic offset overflow propagates as
  `ArithmeticException` before result identity allocation. The attributes themselves can
  represent `Long.MAX_VALUE`; the input-dependent Tensor expression is where checked result
  geometry is enforced.
- Zero-width padding and repeat-one tiling still produce distinct explicit expression identities
  with unresolved layouts. Scalar calls use empty arrays and are valid.
- `boolInput.pad(before, after, ScalarValue.bool(false))` is valid. Passing a FLOAT64 value to that
  receiver fails rather than applying truthiness conversion.
- Identifier exhaustion can occur only at final derived Tensor construction, after all local
  immutable metadata has been prepared.

### Tensor composition expressions

The current API provides static `Tensor.concat(int, Tensor...)`, static
`Tensor.stack(int, Tensor...)`, and instance `tensor.unstack(int)`. Concat and stack snapshot
ordered non-empty inputs, validate their existing or insertion axis and operation-specific exact
type/Shape rules, and return fresh unresolved-layout results with exact ordered provenance.

Unstack is an ordered convenience over scalar selection. It first normalizes an existing axis,
requires a static selected extent no larger than `Integer.MAX_VALUE`, and validates the complete
result count before creating any output. It then returns the immutable list equivalent to:

```java
List.of(input.select(axis, 0), input.select(axis, 1), /* ... */)
```

For input Shape `[2, 3, 4]` and axis `1`, the three outputs correspond to `SELECT(1, 0)`,
`SELECT(1, 1)`, and `SELECT(1, 2)`, each with Shape `[2, 4]`. Each output is a fresh,
unlabeled, storage-free Tensor with its own one-output `TensorProducer`,
`TensorProvenance.outputIndex() == 0`, and exact input `[input]`. The outputs do not share a
producer. Each also uses scalar select's layout rule: resolved non-empty input geometry can produce
a selected-stride/advanced-offset logical view, while unresolved input geometry or an empty result
Shape remains unresolved.

A zero selected extent returns the canonical immutable empty list without creating an operation,
producer, Tensor, or identifier. If identifier exhaustion occurs during a non-empty request,
earlier identifiers remain consumed, the exception propagates, and no partial list is returned.

This provenance shape intentionally differs from a future genuine multi-output operation such as
top-K. Unstack is N independently reproducible SELECT occurrences, so it has N producers and every
output index is zero. A future top-K values/indices pair would be one semantically indivisible
occurrence with one shared producer and distinct output indices. This comparison preserves the
shared multi-output provenance contract; it does not specify a top-K API.

#### Complete tensor-composition expression example

##### Goal and inputs

Construct CONCAT and STACK expressions, then unstack a static axis as repeated SELECT.

```java
Tensor first = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(2, 1), Optional.empty(), false));
Tensor second = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(2, 2), Optional.empty(), true));

Tensor concatenated = Tensor.concat(-1, first, second);
Tensor stacked = Tensor.stack(0, first, first);
List<Tensor> outputs = concatenated.unstack(1);
```

##### Result and interpretation

`concatenated` has Shape `[2, 3]`; `stacked` has Shape `[2, 2, 1]`; and `outputs`
contains three Shape-`[2]` tensors. Output `i` records `SelectKind.SELECT` with
`SelectAttrs(1, i)`, one independent producer, and provenance output index zero. The result
demonstrates model metadata construction, not concatenated or selected values, gradients,
materialization, compiler capture, backend lowering, or execution.

##### Failures and useful variations

Concat and stack reject null or empty input containers and indexed null elements before descriptor
inspection. Unstack rejects an invalid axis, dynamic selected extent, or count above
`Integer.MAX_VALUE` before output allocation. A zero count returns an immutable empty list
without consuming an identifier.
### Window-transform expressions

The current public API constructs three storage-free sliding-window expressions:

```java
tensor.unfold(int axis, long size, long step)
tensor.unfold2d(Window2dAttrs window)
tensor.fold2d(Shape outputShape, Window2dAttrs window)
```

`unfold` requires rank at least one and normalizes its raw axis against the input rank. The selected
dimension must be static, `size` and `step` must be positive, and `size` must fit that dimension.
For selected extent `D`, it replaces that extent with
`floor((D - size) / step) + 1`, preserves every unaffected exact Dimension reference, and appends
`size` as the final axis. It accepts every current data type. Thus input Shape `[2, 5, 3]`, axis
`1`, size `3`, and step `1` produces `[2, 3, 3, 3]`.

`unfold2d` requires rank-four NCHW input and a floating data type. Batch may be dynamic and is
retained exactly; channel, height, and width must be static. `fold2d` requires floating rank-three
canonical columns and an explicit rank-four NCHW output Shape. Column and output batch Dimensions
must be structurally equal, and output channel, height, and width must be static. For each spatial
dimension, both forms use checked `long` arithmetic:

```text
effectiveKernel = dilation * (kernel - 1) + 1
numerator       = input + 2 * padding - effectiveKernel
positions       = floor(numerator / stride) + 1       in floor mode
positions       = ceil(numerator / stride) + 1        in ceil mode
```

The effective kernel must fit the padded input. Ceil mode uses quotient and remainder rather than
the overflow-prone `numerator + stride - 1` identity. `unfold2d` produces canonical Shape
`[N, C * kernelHeight * kernelWidth, outputHeight * outputWidth]`; `fold2d` requires its column
dimensions to match that same checked geometry and retains the exact supplied output Shape.
Conceptual `[1, 1, 3, 3]` input with a 2-by-2 kernel, unit stride and dilation, zero symmetric
padding, and floor mode unfolds to `[1, 4, 4]`. Folding compatible columns to `[1, 1, 3, 3]`
scatter-adds four contributions at the center and one at each corner, without overlap averaging.

All three results preserve the exact input data type and gradient-eligibility flag, use unresolved
layout, record the matching normalized attributes and exact `[input]` provenance, and receive a
fresh unlabeled identity. They do not inspect input layout, label, provenance, storage, or values;
attach or allocate storage; materialize windows; accumulate overlaps; create a gradient rule;
capture a graph; lower to a backend; or execute.

#### Complete window-transform expression example

This runnable example observes only Shape, operation, provenance, and storage metadata:

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class TensorWindowExpressionExample {
    private static Tensor tensor(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));
    }

    public static void main(String[] args) {
        Tensor unfolded = tensor(Shape.of(2, 5, 3)).unfold(1, 3, 1);
        Window2dAttrs window = new Window2dAttrs(
                2, 2, 1, 1, 0, 0, 1, 1, false);
        Tensor columns = tensor(Shape.of(1, 1, 3, 3)).unfold2d(window);
        Tensor image = columns.fold2d(Shape.of(1, 1, 3, 3), window);

        System.out.println(unfolded.descriptor().shape());
        System.out.println(columns.descriptor().shape());
        System.out.println(image.descriptor().shape());
        System.out.println(unfolded.provenance().orElseThrow().operation().kind());
        System.out.println(image.provenance().orElseThrow().operation().kind());
        System.out.println(image.descriptor().requiresGrad());
        System.out.println(image.descriptor().layout().isEmpty()
                && image.hostStorage().isEmpty());
    }
}
```

It prints:

```text
Shape[2, 3, 3, 3]
Shape[1, 4, 4]
Shape[1, 1, 3, 3]
UNFOLD_AXIS
FOLD2D
true
true
```

Invalid rank, axis, intrinsic sign, required staticity, window fit, type, or column compatibility
fails before identity allocation. Checked geometry overflow propagates as `ArithmeticException`.
Null references are rejected in parameter order. Identifier exhaustion remains possible only at
final derived-Tensor construction after all local immutable metadata has been prepared.

### Matrix-multiplication expressions

`Tensor.matmul(right)` constructs one storage-free expression for matrix multiplication, written
mathematically as `left @ right`. A **contraction dimension** is the shared extent whose pairwise
products are summed. For rank-two-or-greater operands, the last two axes are matrix axes: the
left Shape ends in `[M, K]`, the right Shape ends in `[K, N]`, and `K` is the contraction
dimension. Earlier axes are **batch dimensions** and broadcast right-aligned.

A rank-one left operand is treated as `[1, K]` for the operation and its inserted row axis is
removed from the result. A rank-one right operand is treated as `[K, 1]` and its inserted column
axis is removed. This gives the complete rank/Shape relationship:

| Left Shape | Right Shape | Result Shape | Interpretation |
|---|---|---|---|
| `[K]` | `[K]` | `[]` | vector dot product, returned as a scalar |
| `[M, K]` | `[K]` | `[M]` | matrix times vector |
| `[K]` | `[K, N]` | `[N]` | vector times matrix |
| `[M, K]` | `[K, N]` | `[M, N]` | matrix times matrix |
| `[2, M, K]` | `[1, K, N]` | `[2, M, N]` | batched multiplication with singleton expansion |
| `[K]` | `[B, K, N]` | `[B, N]` | vector broadcast over a right batch |
| `[B, M, K]` | `[K]` | `[B, M]` | right vector broadcast over a left batch |

Read each row by first broadcasting the leading batch prefixes, then appending `M` unless the
left operand is rank one and appending `N` unless the right operand is rank one. Every retained
row, column, equal batch, singleton-opposing, or unpaired batch Dimension is the exact input
reference.

#### Concrete matrix example

The input values and Shapes are:

```text
left  Shape [2, 3] = [[1, 2, 3],
                      [4, 5, 6]]

right Shape [3, 2] = [[1, 2],
                      [3, 4],
                      [5, 6]]
```

Each output position is one left-row/right-column dot product:

```text
result[0,0] = 1*1 + 2*3 + 3*5 = 22
result[0,1] = 1*2 + 2*4 + 3*6 = 28
result[1,0] = 4*1 + 5*3 + 6*5 = 49
result[1,1] = 4*2 + 5*4 + 6*6 = 64
```

Therefore the mathematical result has Shape `[2, 2]` and values:

```text
[[22, 28],
 [49, 64]]
```

This calculation explains the recorded operation meaning. Calling `matmul` does not read the
input values or produce these values eagerly; the current model creates metadata only.

#### Dynamic dimensions and local failures

Both ranks must be at least one. Two unequal static contraction dimensions fail immediately.
When either contraction dimension is unresolved, equality is deferred because it does not affect
the exact result Shape and both input descriptors remain in provenance. Batch broadcasting is
more restrictive because construction must select an exact output Dimension:

- equal unresolved dimensions retain the exact left reference;
- a static singleton retains the exact opposing reference;
- an unpaired leading batch dimension retains its exact input reference;
- an unresolved dimension paired with static non-singleton `S` selects that exact static result
  Dimension and defers the obligation that the unresolved extent bind to `1` or `S`;
- unequal unresolved batch dimensions fail because no exact result Dimension is locally
  derivable; and
- unequal static non-singleton batch dimensions fail.

These rules are local to MATMUL and do not change general `ShapeBroadcast`. Null checks, numeric
promotion, rank checks, static contraction checks, and leading-to-trailing batch checks all occur
before result identity allocation. Later compiler validation or concrete binding must prove every
deferred equality or singleton-or-equal obligation before execution.

#### Data type, numerical meaning, and result metadata

`DataTypePromotion.promoteNumeric` accepts same-category floating pairs
(`BFLOAT16`, `FLOAT32`, `FLOAT64`) and same-category signed-integral pairs (`INT32`, `INT64`).
The promoted type is the result type. BOOL and floating/integral pairs fail, and construction
inserts no cast.

For a `FLOAT64` result, pairwise products accumulate in FLOAT64. `FLOAT32` and `BFLOAT16` results
accumulate in FLOAT32, with the latter converted to BFLOAT16 output. Floating implementations may
reassociate terms and use fused multiply-add, so no traversal order, bitwise result, or identical
cross-backend rounding is promised. IEEE-754 special values follow the selected multiply and add
operations; an empty contraction produces positive floating zero. Integral products and sums use
the promoted signed width modulo `2^32` or `2^64`; an empty integral contraction produces zero.

Every valid call returns a fresh, unlabeled Tensor with the promoted type, exact derived Shape,
unresolved layout, no storage, and gradient eligibility equal to the logical OR of the operands.
Its one-output provenance has index zero and retains `MatmulKind.MATMUL`,
`NoOperationAttrs.INSTANCE`, and exact ordered inputs `[left, right]`. This metadata records a
gradient request, not a gradient rule. It neither captures or compiles a graph nor promises
compiler validation, backend lowering, a kernel, storage allocation, or execution.

### Matrix-multiplication semantic kind

`MatmulKind` contains exactly `MATMUL`. The kind pairs only with
`NoOperationAttrs.INSTANCE`; its family-owned `OperationSignature` fixes two logical inputs and
one logical output. It records backend-independent vector, matrix, and batched matrix
multiplication meaning, including the accumulation policies above. The kind stores no operands,
Shapes, deferred constraints, result descriptor, provenance, algorithm, gradient, compiler,
backend, or execution state. `Tensor.matmul` separately owns input-aware local validation and
expression construction.

### Stable sort and argsort expressions

`input.sort(axis)` and `input.argsort(axis)` request ascending stable ordering along one logical
axis. Their two-argument overloads select descending non-NaN order when `descending` is true.
Every current data type is eligible: signed integers use signed numerical order, BOOL uses
`false < true`, and floating values use numerical order with the additional rules below.

For each independent slice along the normalized axis:

- equal finite, integral, or BOOL values keep increasing original logical indices;
- negative zero precedes positive zero ascending and follows it descending;
- negative and positive infinity participate in ordinary numerical order;
- all NaNs remain after every non-NaN in both directions; multiple NaNs keep input order; and
- `sort` retains the selected input representation, including signed zero and NaN payload bits.

The following value table shows the requested result for one conceptual FLOAT64 row. It describes
eventual ordering semantics; current model construction does not evaluate the row.

| Request | Values | Logical indices |
|---|---|---|
| Input | `[3.0, NaN, -0.0, +0.0, 3.0]` | `[0, 1, 2, 3, 4]` |
| Ascending | `[-0.0, +0.0, 3.0, 3.0, NaN]` | `[2, 3, 0, 4, 1]` |
| Descending | `[3.0, 3.0, +0.0, -0.0, NaN]` | `[0, 4, 3, 2, 1]` |

#### Complete metadata example

The example constructs both requests and inspects facts that the current model actually records.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Optional;

Tensor input = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT64, Shape.of(5), Optional.empty(), true));
Tensor values = input.sort(-1);
Tensor indices = input.argsort(0, true);

System.out.println(values.descriptor().dataType());
System.out.println(indices.descriptor().dataType());
System.out.println(values.provenance().orElseThrow().operation());
System.out.println(indices.provenance().orElseThrow().operation());
```

The output metadata is FLOAT64, INT64, `Operation[kind=SORT,
attrs=SortAttrs[axis=0, descending=false]]`, and `Operation[kind=ARGSORT,
attrs=SortAttrs[axis=0, descending=true]]`. The negative axis was normalized before attributes
construction. This proves type and semantic provenance; it does not calculate the table's values.

Each successful call creates a fresh, independent one-input, one-output producer at output slot
zero. Both results retain the exact input Shape reference and leave layout unresolved. `sort`
also preserves input type and gradient eligibility; `argsort` always uses INT64 and false gradient
eligibility. Results are unlabeled and storage-free. Static empty and singleton selected extents
are valid, as are dynamic extents, because the Shape does not change. A scalar has no valid axis.

`OrderingKind` contains exactly `SORT` and `ARGSORT`. Each kind requires `SortAttrs`, one input,
and one output. `SortAttrs(axis, descending)` holds an already normalized non-negative axis and
the direction flag; stability and NaN-last placement are fixed semantics, not attributes.
Construction checks only metadata and creates provenance. It neither compares values nor selects
an algorithm, gradient rule, compiler behavior, backend route, runtime behavior, or execution
support. A future top-K capability is separate because it changes Shape and needs one shared
values-and-indices producer.

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
| `OperationSignature` | An implemented immutable structural contract that pairs one exact attributes class with inclusive input/output-count bounds. |
| `Operation` | An implemented immutable descriptor that stores one kind and one compatible `OperationAttrs` value. |
| `BinaryArithmeticKind` | The implemented production enum for seven parameterless tensor-to-tensor arithmetic meanings. |
| `BinaryComparisonKind` | The implemented production enum for six parameterless ordered tensor-to-tensor comparison meanings. |
| `BooleanLogicalKind` | The implemented production enum for parameterless elementwise boolean conjunction, disjunction, and negation meanings. |
| `WhereSelectionKind` | The implemented production enum for one parameterless ternary conditional-selection meaning. |
| `CastKind` | The implemented production enum for one parameterized elementwise data-type conversion meaning. |
| `CastAttrs` | The implemented immutable value for one exact non-null target `DataType`. |
| `AggregateReductionKind` | The implemented production enum for seven ordinary aggregates, five floating advanced/statistical reductions, and axis-only arg-min/arg-max. |
| `AxisReductionAttrs` | The implemented immutable normalized-axis and retained-dimension value for ordinary single-axis reductions. |
| `MultiAxisReductionAttrs` | The implemented immutable caller-ordered distinct normalized axes and retained-dimension value for ordinary, log-sum-exp, and norm reductions. |
| `StatisticalReductionAttrs` | The implemented immutable ordered axes, retained-dimension choice, and non-negative correction for variance/standard deviation. |
| `ArgExtremaAttrs` | The implemented immutable normalized-axis, retained-dimension, and explicit tie-policy value shared by arg-min and arg-max. |
| `ArgExtremaTiePolicy` | The implemented `FIRST_INDEX` or `LAST_INDEX` choice for equal extrema. |
| `MaskedReductionAttrs` | The implemented immutable normalized reduction axis for first-class, two-input masked sum and mean semantics. |
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
| `SliceKind` | The implemented production enum whose sole `SLICE` value identifies finite signed-step logical coordinate selection. |
| `SliceAttrs` | The implemented immutable normalized starts, selected lengths, distinct axes, and signed non-zero steps for a slice. |
| `PadKind` | The implemented production enum whose sole `PAD` value identifies constant padding around every input axis. |
| `PadAttrs` | The implemented immutable ordered before/after widths and exact typed padding constant. |
| `TileKind` | The implemented production enum whose sole `TILE` value identifies complete-pattern per-axis tiling. |
| `TileAttrs` | The implemented immutable ordered positive complete-pattern repeat counts. |
| `TensorCompositionKind` | The implemented production enum for ordered `CONCAT` and inserted-axis `STACK` meanings. |
| `CompositionAxisAttrs` | The implemented immutable normalized non-negative axis shared by concat and stack. |
| `SelectKind` | The implemented production enum whose sole `SELECT` value fixes one scalar coordinate on one source axis and removes that axis. |
| `SelectAttrs` | The implemented immutable normalized non-negative source axis and scalar coordinate for `SELECT`. |
| `AxisGatherKind` | The implemented production enum for canonical Gather and aligned same-rank Gather Elements meanings. |
| `IndexAxisAttrs` | The implemented immutable normalized non-negative data-axis value shared by Gather and Gather Elements. |
| `OneHotKind` | The implemented production enum whose sole `ONE_HOT` value identifies dense trailing-axis BOOL index encoding. |
| `OneHotAttrs` | The implemented immutable positive static depth of one-hot's appended class axis. |
| `AxisScatterKind` | The implemented production enum whose sole `SCATTER_ELEMENTS` value identifies configurable same-rank functional scatter. |
| `ScatterReduction` | The implemented reusable replacement, addition, multiplication, maximum, or minimum meaning for configurable functional scatter. |
| `ScatterElementsAttrs` | The implemented immutable normalized axis and explicit non-null reduction for `SCATTER_ELEMENTS`. |
| `UnaryElementwiseKind` | The implemented production enum for nineteen parameterless unary numeric and activation meanings. |
| `FloatingClassificationKind` | The implemented production enum for three parameterless floating value classifications with fixed BOOL results. |
| `ScalarElementwiseKind` | The implemented production enum for eight parameterized one-input scalar elementwise meanings. |
| `ScalarValueAttrs` | The implemented immutable holder for one exact typed scalar parameter. |
| `ClampRangeAttrs` | The implemented immutable value for exact same-type numeric inclusive clamp bounds. |

`OperationKind` declares `name()` and `signatures()`. The name is stable diagnostic text, not a
serialization token or global lookup key. Each concrete family owns a stable immutable non-empty
list of signatures, so adding a family does not require a central registry or a root switch over
all kinds. `signatureFor(attrs)` matches the exact runtime attributes class and rejects missing,
duplicate, null, or incompatible declarations. Two values from different kind families remain
different typed values even if both names contain the same text.

An `OperationSignature` describes one structurally valid occurrence variant. Its inclusive
minimum and maximum input/output counts describe logical positions, not Tensor rank or element
count. Equal minimum and maximum values form a **fixed** cardinality. Different finite bounds form
a **bounded** cardinality. `Integer.MAX_VALUE` is the inclusive upper bound used for an
effectively **variadic** input list. Signatures deliberately do not validate operand data types,
Shapes, layouts, numerical policy, backend support, or executability.

`OperationAttrs` deliberately declares no methods. A family-specific implementation defines its
own typed fields and must be immutable, defensively isolate mutable constructor inputs, and provide
structural equality and hashing. `NoOperationAttrs.INSTANCE` makes the absence of parameters an
explicit non-null value rather than `null` or an empty map. The marker cannot enforce immutability,
but `Operation` now validates that the exact concrete attributes class is explicitly accepted by
the selected kind before retaining both references.

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
broadcasting, result-descriptor construction, and ordered provenance. ADD, SUB, MUL, and DIV are
ordinary ordered IEEE-754 requests in the result type. MIN and MAX propagate NaN, order infinities
normally, and choose negative or positive zero, respectively, from opposite signed zeros. No NaN
payload, intermediate precision, exact instruction, bitwise result, gradient rule, compiler
capture, execution, or backend availability is promised here. The inherited enum names and text
are stable diagnostics, not serialization
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
a Tensor result, provenance value, graph occurrence, or executable operation. Construction
enforces the exact `NoOperationAttrs` pairing; the family signature declares two inputs and one
output.

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

The resulting operation retains the exact `AND` kind and canonical no-attributes value. Family
signatures declare two inputs for `AND` and `OR`, one input for `NOT`, and one output for each.
`CompiledNode` validates those local occurrence counts. The enum stores no input references and
creates no Tensor, provenance, or result descriptor.

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

The resulting operation retains the exact `WHERE` kind and canonical no-attributes value. Its
family signature declares three ordered inputs and one output and enforces the exact attributes
pairing. The enum itself stores no input objects.

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

The resulting descriptor retains the exact kind and attributes values. `Operation` construction
enforces the exact `CAST`/`CastAttrs` pairing, and the signature declares one input and one output.
A null target fails during `CastAttrs` construction with
`NullPointerException` and message `targetDataType`.

The current `Tensor.cast(DataType)` method separately owns source-descriptor inspection,
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
| `ARG_MIN` | Select a logical index of a minimum value along one axis. | One normalized axis with an explicit tie policy. |
| `LOG_SUM_EXP` | Compute log of summed exponentials. | Ordered distinct normalized axes. |
| `VARIANCE` | Compute corrected second central moment. | Ordered distinct normalized axes plus correction. |
| `STANDARD_DEVIATION` | Compute non-negative square root of corrected variance. | Ordered distinct normalized axes plus correction. |
| `L1_NORM` | Sum absolute values. | Ordered distinct normalized axes. |
| `L2_NORM` | Compute square root of summed squares. | Ordered distinct normalized axes. |

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
and the current `argMin`/`argMax` expressions normalize a caller axis through the input Shape before
creating attributes.

When `keepDimensions` is false, the selected axis is removed from the eventual result. When it is
true, the axis remains with extent one. The record stores that request but does not construct an
output Shape. Full reductions in this contract have no `keepDimensions` parameter, and
`ARG_MIN`/`ARG_MAX` have no full form.

Arg-min and arg-max pair only with shared
`ArgExtremaAttrs(axis, keepDimensions, tiePolicy)`. `FIRST_INDEX` requests the smallest logical
index among equal extrema along the selected axis; `LAST_INDEX` requests the largest. A logical
index is an axis position rather than a storage offset. The policy must be supplied explicitly and
is never defaulted by the semantic value:

```java
Operation argMax = new Operation(
        AggregateReductionKind.ARG_MAX,
        new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.FIRST_INDEX));
```

`ArgExtremaAttrs` rejects a negative axis before checking the tie policy and rejects a null policy
with `NullPointerException("tiePolicy")`. Both attribute records use generated record value
semantics, while their text remains diagnostic rather than serialization or dispatch syntax.
Family-owned signatures enforce these exact attribute variants. Ordinary forms declare one input;
masked SUM and MEAN declare the ordered two-input occurrence `[input, mask]`; every variant has one
output.

A masked sum or mean pairs its existing kind with `MaskedReductionAttrs(axis)`. The axis is
already normalized, non-negative, and removed from the eventual result. The attributes store no
mask, Shape, mapping, or broadcast plan. `Integer.MAX_VALUE` remains structurally valid because
the attribute cannot know the eventual input rank. Public `Tensor.sum(axis, mask)` and
`Tensor.mean(axis, mask)` separately normalize the axis and require one ordinary right-aligned
broadcast whose result equals the input Shape before constructing the attributes.

The public provenance order is `[input, mask]`. False mask positions exclude their aligned input
values, including NaN and infinity. When no values are selected, a masked sum produces zero. A
masked mean divides by the true selected-count for each output and produces NaN when that count is
zero, without promising a NaN payload or bit pattern. These are semantic
requirements for later execution; constructing `MaskedReductionAttrs` does not inspect a mask,
count values, divide, derive an output descriptor, or execute computation. The signature matrix
accepts `MaskedReductionAttrs` only for `SUM` and `MEAN`.

Multi-axis ordinary, log-sum-exp, and norm occurrences use
`MultiAxisReductionAttrs(axes, keepDimensions)`. Variance and standard deviation use
`StatisticalReductionAttrs(axes, keepDimensions, correction)`. Both snapshot ordered distinct
normalized axes; the statistical value additionally rejects negative correction. An empty axis
list means a point domain, not full reduction.

The public Tensor surface now covers every current aggregate semantic kind. `Tensor.sum`, `prod`,
reduction `min`, and reduction `max` provide floating/integral full and single-axis expression
construction; `mean` remains floating-only. Public `Tensor.all` and `Tensor.any` provide the
corresponding exact-BOOL forms with fixed false gradient eligibility. Public `Tensor.argMin` and
`Tensor.argMax` provide axis-only floating/integral construction with fixed INT64, false-gradient
results and explicit tie semantics. Twenty-six multi-axis/statistical methods add ordered axes,
optional retention, and population-default/explicit correction. Every current family derives
Shapes locally and records exact one-input provenance. Integral ordinary reductions use exact-type modular sum/product, signed
minimum/maximum, and bounded empty identities. Arg extrema use shared NaN, signed-zero, infinity,
tie, and invalid-empty-selected-axis policy. Aggregate extrema remain typed separately from equally
named binary elementwise kinds, while aggregate ALL/ANY remain typed separately from elementwise
AND/OR. These contracts define requested meaning only and do not evaluate values, implement a
gradient or numerical algorithm, capture a graph, lower operations, report backend availability,
or execute.

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
normalized axis zero. Its signature enforces the exact attributes class and declares one input and
one output, but it does not validate axis bounds for a particular shape or result facts.
`CumulativeSumAttrs` accepts every non-negative `int`, including `Integer.MAX_VALUE`, and
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
`Operation` retains either kind and the exact attributes reference; their shared signature
enforces the attributes class and one-input, one-output cardinality, but not rank bounds or result
facts.

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

The resulting `Operation` retains the exact kind and canonical no-attributes singleton. Its
signature enforces that pairing and declares one input and one output.

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

Both operations retain the exact `attrs` reference. Their shared family signature enforces
`TargetShapeAttrs` and declares one input and one output.
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
`Operation` retains the supplied kind and attributes, while family signatures enforce the exact
permutation-versus-axis attributes variants. Public `Tensor.permute` and `transpose` now construct result descriptors, logical view
layout when input geometry is resolved, and exact one-input provenance. Public `expandDims` and
`squeeze` now construct the corresponding rank-edited result descriptors, conditional logical
view layouts, and exact one-input provenance. Gradients, compiler behavior, materialization,
backend support, and execution remain planned.

### Slice semantic kind and normalized attributes

The public enum `io.github.pho001.synaptik.model.operation.layout.SliceKind` implements
`OperationKind` with exactly one constant, `SLICE`. It identifies a one-input, same-rank logical
selection. `SliceAttrs(starts, lengths, axes, steps)` carries four equal-size parallel lists. At
entry `i`, it selects exactly `lengths[i]` coordinates beginning at inclusive `starts[i]` and
advancing by signed non-zero `steps[i]` on normalized input axis `axes[i]`. Coordinate `k` is:

```text
starts[i] + k * steps[i], for 0 <= k < lengths[i]
```

An axis without an entry retains its complete logical coordinate range.

#### Parallel half-open example

##### Goal and inputs

Describe a conceptual mixed-direction slice of Shape `[3, 5]` that keeps all three rows and
reverses the columns. The already normalized parameters are starts `[0, 4]`, lengths `[3, 5]`,
axes `[0, 1]`, and steps `[1, -1]`.

```java
SliceAttrs attrs = new SliceAttrs(
        List.of(0L, 4L),
        List.of(3L, 5L),
        List.of(0, 1),
        List.of(1L, -1L));
Operation slice = new Operation(SliceKind.SLICE, attrs);
```

##### Meaningful entries and result

- Entry zero starts at zero, has length three, and steps by one, so its coordinates are rows
  `0`, `1`, and `2`.
- Entry one starts at four, has length five, and steps by negative one, so its coordinates are
  columns `4`, `3`, `2`, `1`, and `0`.
- The two entries apply in parallel. `slice.kind()` is exactly `SliceKind.SLICE`, and
  `slice.attrs()` is exactly the supplied `attrs` reference.

The conceptual selected coordinate grid therefore contains all three rows with reversed column
order. This result explains the requested logical selection; constructing the attributes and
`Operation` does not construct a Tensor, calculate a result Shape or layout, read values, or
execute work.

##### Ownership, validation, and useful variations

Starts, lengths, and steps use `long`, matching the numeric width of Shape dimensions and layout
geometry; axes use `int`, matching Java rank and axis positions. Construction requires non-null
equal-size lists and elements, rejects negative starts, lengths, or axes, repeated axes, and zero
steps. Length zero requires canonical start zero. For positive length, checked arithmetic verifies
that the final selected coordinate is representable and non-negative. This admits
`Long.MIN_VALUE` as a step for a one-coordinate sequence. Validation completes before
`List.copyOf` stores one immutable snapshot per component. Caller mutation therefore cannot change
the attributes, accessor mutation fails, and entry order plus all values define equality.

Four empty lists form a normalized identity slice that constrains no axes. `SliceAttrs` has no
input Shape, so a non-negative axis may still exceed a future rank. It stores neither raw bounds
nor a direction-dependent end sentinel. Raw negative axes or bounds, clamping, selected-static-
dimension checks, checked length calculation, result Shape and layout, and provenance are supplied
by the current public expression boundary described under [Slice expressions](#slice-expressions).

The current single-axis convenience uses the same semantic kind with one step-one entry:

```java
new SliceAttrs(
        List.of(normalizedStart),
        List.of(selectedLength),
        List.of(normalizedAxis),
        List.of(1L));
```

The explicit-step single-axis method stores its signed step, and `flip` stores one negative-step
entry per requested axis in the same `SliceAttrs`. Neither is a second kind such as `SLICE_AXIS`
or `FLIP`. These semantic values also define no storage view, materialization, gradient or
backward scatter, compiler canonicalization, ONNX mapping, backend route, or execution behavior.

### Pad and tile semantic kinds and normalized attributes

The public enums `io.github.pho001.synaptik.model.operation.layout.PadKind` and
`io.github.pho001.synaptik.model.operation.layout.TileKind` implement `OperationKind` with exactly
`PAD` and `TILE`, respectively. `PAD` means adding constant-filled logical positions before and
after every input axis without changing rank. `TILE` means repeating the complete input pattern a
positive number of times along every axis without changing rank. The kinds identify meaning only;
they do not inspect an input or calculate values or extents.

`PadAttrs(before, after, constantValue)` stores two ordered equal-size `List<Long>` width lists and
one raw Java `double`. Entry `i` in each list applies before or after the same normalized axis
position. `TileAttrs(repeats)` stores one ordered `List<Long>` whose entry `i` is the number of
complete input-pattern repetitions along normalized axis position `i`.

#### Constant-padding example

##### Goal and inputs

Describe one-dimensional constant padding for conceptual input `[10, 20]`: add one position
before, two positions after, and fill those logical positions with `-1`.

```java
PadAttrs attrs = new PadAttrs(List.of(1L), List.of(2L), -1.0d);
Operation padded = new Operation(PadKind.PAD, attrs);
```

##### Result and interpretation

The request means `[-1, 10, 20, -1, -1]`. `padded.kind()` is exactly `PadKind.PAD`, and
`padded.attrs()` is exactly the supplied `attrs` reference. This is a conceptual semantic result:
construction does not create a Tensor, populate values, derive a Shape or layout, convert `-1.0d`
to an input `DataType`, or execute padding.

#### Complete-pattern tiling example

##### Goal and inputs

Describe tiling conceptual input `[[1, 2], [3, 4]]` twice along axis zero and three times along
axis one.

```java
TileAttrs attrs = new TileAttrs(List.of(2L, 3L));
Operation tiled = new Operation(TileKind.TILE, attrs);
```

##### Result and interpretation

The complete row pattern repeats three times and the complete two-row pattern repeats twice:

```text
[[1, 2, 1, 2, 1, 2],
 [3, 4, 3, 4, 3, 4],
 [1, 2, 1, 2, 1, 2],
 [3, 4, 3, 4, 3, 4]]
```

This is complete-pattern tiling, not scalar-element repeat: it does not transform `[1, 2]` into
`[1, 1, 1, 2, 2, 2]`. `tiled.kind()` is exactly `TileKind.TILE`, and `tiled.attrs()` is exactly
the supplied `attrs` reference. The example states requested logical meaning without claiming
Tensor construction, value materialization, or execution.

##### Ownership, validation, and useful variations

Both records validate caller-owned lists in ascending index order and copy only after every
element passes. The stored immutable snapshots preserve order and values but not caller list
identity; later caller mutation cannot change them, and accessor mutation fails. Empty pad lists
and an empty repeat list are valid rank-zero scalar identity parameters. `Long.MAX_VALUE` is also
structurally valid because these records perform no input-rank matching or result-Shape arithmetic.

`PadAttrs` validation order is exact:

1. Null-check `before`, then `after`, producing `NullPointerException` with the component name.
2. Reject unequal sizes with `IllegalArgumentException("before and after must have matching sizes")`.
3. At each ascending index, null-check `before[i]`, then `after[i]`, using that indexed message.
4. Reject a negative before width with `before[i] must be non-negative: value`, then a negative
   after width with `after[i] must be non-negative: value`.
5. Snapshot `before`, then `after` exactly once each.

Every `constantValue` is retained bit-for-bit as the supplied primitive, including positive and
negative zero, finite values, NaN payloads, and either infinity. No finiteness, range, rounding,
saturation, BOOL normalization, or `DataType` compatibility rule is applied. Ordinary Java record
and `double` equality/hash semantics still apply; raw-bit retention is not a custom equality or
serialization contract.

`TileAttrs` first null-checks `repeats`, then visits entries in ascending order. A null entry fails
with `NullPointerException("repeats[i]")`; a zero or negative entry fails with
`IllegalArgumentException("repeats[i] must be positive: value")`. Only then does construction take
its single immutable snapshot. Result-Shape multiplication and overflow checks are deferred.

Family signatures enforce `PAD` with `PadAttrs` and `TILE` with `TileAttrs`, each with one input
and one output. Neither family defines public Tensor request syntax, result inference, layout or
storage behavior, materialization, provenance, gradients, compiler or planning behavior, backend
or ONNX behavior, or execution.

### Tensor composition semantic kinds and attributes

`TensorCompositionKind` implements `OperationKind` with exactly `CONCAT` and `STACK`, in
that order. Both pair with `CompositionAxisAttrs` and accept one or more ordered inputs with
exactly one output. CONCAT joins along an existing normalized input axis and preserves rank; STACK
joins same-shaped inputs along a newly inserted normalized result axis and increases rank by one.
The semantic types do not validate input rank, type, Shape, or result metadata.

Unstack is not a `TensorCompositionKind` value and has no dedicated attributes. Public
`Tensor.unstack` is the repeated-SELECT convenience described under
[tensor composition expressions](#tensor-composition-expressions). This keeps each selection a
complete independent occurrence and avoids confusing a convenience list with one multi-output
operation.

### Scalar select semantic kind and attributes

The public enum `io.github.pho001.synaptik.model.operation.index.SelectKind` implements
`OperationKind` with exactly one constant, `SELECT`. It identifies scalar-index selection: fix
one coordinate on one existing source axis and remove that axis from the logical result.
`SelectAttrs(axis, index)` carries the normalized zero-based source-axis position as an `int` and
the normalized zero-based scalar coordinate as a `long`.

#### Scalar-axis removal example

##### Goal and inputs

Describe selecting coordinate `2` on axis `1` from a conceptual source with Shape `[2, 3, 4]`.

```java
SelectAttrs attrs = new SelectAttrs(1, 2L);
Operation select = new Operation(SelectKind.SELECT, attrs);
```

##### Result and interpretation

Axis `1` has extent `3`. Fixing its coordinate at index `2` leaves axes `0` and `2`, so the
conceptual result Shape is `[2, 4]`. `select.kind()` is exactly `SelectKind.SELECT`, and
`select.attrs()` is exactly the supplied `attrs` reference.

The example explains semantic meaning only. Constructing these values does not inspect an input,
derive a result Shape or layout, construct a Tensor or provenance, read a value, or execute work.

##### Validation and boundaries

`SelectAttrs` checks the axis first and rejects a negative value with
`IllegalArgumentException("axis must be non-negative: <axis>")`. Once the axis is valid, it
rejects a negative index with
`IllegalArgumentException("index must be non-negative: <index>")`. Zero, `Integer.MAX_VALUE`,
and `Long.MAX_VALUE` are structurally valid, and both primitive values are retained unchanged.

The attributes store no input Shape, rank, or selected-axis extent. Structural validity therefore
does not prove that the axis exists or that the index is in bounds. The family signature enforces
`SELECT` with `SelectAttrs` and declares one input and one output.

`SELECT` differs from conditional `WHERE`, which chooses between branch values position by
position, and general `SLICE`, which selects half-open intervals without removing an axis. It also
differs from gather operations whose indices are tensors rather than one intrinsic scalar
coordinate. Public `Tensor.select` now supplies input-aware axis/index normalization,
validation, result Shape and conditional logical-view layout, and exact one-input provenance as
described under [scalar select expressions](#scalar-select-expressions). Gradients, graph/compiler
capture and canonicalization, materialization, backend lowering, and execution remain planned.

### Gather and Gather Elements semantic kinds and attributes

`AxisGatherKind` implements `OperationKind` with exactly `GATHER` and `GATHER_ELEMENTS`, in
that order. Both consume ordered logical inputs `[data, indices]`, pair with
`IndexAxisAttrs(axis)`, and declare exactly two inputs and one output.

- `GATHER` replaces the selected data axis with the complete indices Shape. Data
  `[2, 3, 4]`, axis `1`, and indices `[5, 6]` conceptually produce `[2, 5, 6, 4]`;
  scalar indices produce `[2, 4]`.
- `GATHER_ELEMENTS` requires same-rank indices aligned with data away from the selected axis and
  conceptually produces the exact indices Shape. Data `[2, 3, 4]`, axis `1`, and indices
  `[2, 7, 4]` produce `[2, 7, 4]`.

`IndexAxisAttrs` rejects a negative normalized axis but contains no data rank or Shape. Public
`Tensor.gather` and `gatherElements` own index-type validation, caller-axis normalization,
Shape checks and derivation, result metadata, and exact `[data, indices]` provenance. Neither
semantic values nor public expression construction inspect index values or bounds, define
gradients, provide compiler behavior, choose a backend, or execute work.

### One-hot semantic kind and attributes

The public enum `io.github.pho001.synaptik.model.operation.index.OneHotKind` implements
`OperationKind` with exactly `ONE_HOT`. It consumes ordered logical input `[indices]`, produces
one dense BOOL output, requires exact `OneHotAttrs`, and declares exactly one input and one output.
`OneHotAttrs(depth)` retains one positive `long` unchanged. It rejects zero or a negative value
with `depth must be positive: <depth>` and accepts every positive value through
`Long.MAX_VALUE` without claiming that later storage can materialize the Shape.

For an index value `i`, output coordinate `j` on the new trailing axis is true exactly when
`i == j`. Eventual execution requires `0 <= i < depth`; an invalid index is not a request for
wrapping, clamping, a default row, or an all-false row. The kind and attributes contain no input
Tensor, Shape, axis, on/off values, output type, bounds policy, compiler state, backend support,
storage, or execution state. Public `Tensor.oneHot` separately owns input-type and depth
validation, Shape/descriptor construction, and exact one-input provenance as described under
[one-hot encoding expressions](#one-hot-encoding-expressions).

### Gather-ND semantic kind and attributes

The public enum `io.github.pho001.synaptik.model.operation.index.GatherNdKind` implements
`OperationKind` with exactly one constant, `GATHER_ND`. It consumes ordered logical inputs
`[data, indices]`: `data` supplies values, while `indices` supplies coordinate tuples. The final
indices Dimension is the tuple depth `K`. `GatherNdAttrs(batchDimensions)` stores only the
already normalized, non-negative number `B` of shared leading batch Dimensions.

For data rank `R` and indices rank `Q`, every tuple indexes data axes `[B, B + K)`. Data axes
`[B + K, R)` form the untouched suffix of the selected value. The conceptual result Shape is:

```text
indices.shape[0:Q-1] + data.shape[B+K:R]
```

The first term is equivalently `indices.shape[:-1]`: every indices Dimension except the final
tuple-depth Dimension remains in the result.

#### Tuple-index Shape examples

##### Goal and inputs

Compare zero-batch selection, batched selection, and selection of one complete scalar:

```java
GatherNdAttrs attrs = new GatherNdAttrs(1);
Operation gatherNd = new Operation(GatherNdKind.GATHER_ND, attrs);
```

- Data `[2, 3, 4]`, indices `[5, 2]`, `B=0`, and `K=2` select two data axes per tuple.
- Data `[2, 3, 4]`, indices `[2, 5, 1]`, `B=1`, and `K=1` share the leading extent `2` and select
  one later data axis per tuple.
- Data `[2, 3]`, indices `[2]`, `B=0`, and `K=2` select every data axis with one tuple.

##### Results and interpretation

- The first result is `[5, 4]`: indices prefix `[5]` is followed by untouched data suffix `[4]`.
- The second result is `[2, 5, 4]`: indices prefix `[2, 5]` is followed by suffix `[4]`.
- The third result is canonical scalar Shape `[]`, not `[1]`, because both formula terms are
  empty.

The `Operation` in the code example retains exactly `GatherNdKind.GATHER_ND` and the supplied
`attrs` reference. The current zero-batch `Tensor.gatherNd(indices)` convenience uses
`new GatherNdAttrs(0)` rather than a second kind or a default attributes value.

##### Validation and boundaries

`GatherNdAttrs` rejects a negative count with
`IllegalArgumentException("batchDimensions must be non-negative: <batchDimensions>")`. Zero,
positive values, and `Integer.MAX_VALUE` are structurally valid and retained unchanged. The value
contains no input ranks or Shapes, so it cannot prove that `B` fits the inputs, that leading batch
Dimensions match, that the final indices Dimension supplies a valid tuple depth, or that an index
data type is permitted. The family signature enforces `GatherNdAttrs` and declares two inputs and
one output.

Tuple depth remains the final indices Dimension instead of an attribute because it varies with
each operation occurrence. The current public Tensor boundary validates data and indices ranks,
shared batch Dimensions, tuple depth, index type, and result Shape and records exact provenance as
described under [Gather-ND expressions](#gather-nd-expressions). The semantic values themselves do
none of that input-aware work. Gather-ND is distinct from scalar `SELECT`, which stores one
coordinate; axis gather, which indexes one selected axis; and scatter-ND, which writes or combines
updates. Neither semantic values nor current expression construction define index-value bounds,
a numerical algorithm, gradients, compiler behavior, backend behavior, or execution.

### Scatter Elements semantic kind, reduction, and attributes

`AxisScatterKind` implements `OperationKind` with exactly `SCATTER_ELEMENTS`. It consumes
ordered logical inputs `[data, indices, updates]`, pairs with
`ScatterElementsAttrs(axis, reduction)`, and declares exactly three inputs and one output.

Indices and updates have equal rank and Shape and match data away from the selected axis. The
functional result has the exact data Shape and does not mutate data. For data `[2, 3, 4]`, axis
`1`, and indices/updates `[2, 5, 4]`, update coordinate `[0, 4, 2]` uses its index to select
target `[0, index, 2]`.

`ScatterReduction` contains exactly `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`.
`NONE` means replacement and requires unique targets; the arithmetic values combine the base
and all updates for a target. The vocabulary does not define traversal order, numerical edge
behavior, determinism, atomicity, or a backend algorithm.

`ScatterElementsAttrs` validates the normalized non-negative axis before requiring a non-null
reduction. Public `Tensor.scatterElements` owns caller-axis, type, reduction-eligibility, rank,
Shape, descriptor, and provenance validation. Index bounds and duplicate targets require values
and remain outside model metadata construction. The semantic values define no gradients,
compiler-generated adjoints, materialization, lowering, backend behavior, or execution. Fixed-add
adjoints are not current public operation kinds; task 0023 may later define selected
compiler-generated semantics without restoring public aliases.

### Scatter-ND semantic kind and attributes

The public enum `io.github.pho001.synaptik.model.operation.index.ScatterNdKind` implements
`OperationKind` with exactly one constant, `SCATTER_ND`. Functional Scatter-ND starts from base
`data`, uses coordinate tuples from `indices` to address result targets, applies `updates` through
an explicit `ScatterReduction`, and produces a new result with the exact data Shape. It does not
mutate `data` in place. Its ordered logical inputs are `[data, indices, updates]`.

`ScatterNdAttrs(batchDimensions, reduction)` stores the already normalized non-negative count
`B` of shared leading data and indices batch Dimensions, followed by the explicit non-null
reduction. If data rank is `R`, indices rank is `Q`, and the final indices Dimension has extent
`K`, that extent is the tuple depth. Each tuple indexes data axes `[B, B + K)`, while data axes
`[B + K, R)` form the untouched suffix represented by one update slice. The required updates
Shape is:

```text
indices.shape[0:Q-1] + data.shape[B+K:R]
```

Equivalently, it is `indices.shape[:-1] + data.shape[batchDimensions + K:]`.

#### Tuple-index update Shape examples

These examples are conceptual; the current semantic values do not inspect operands or construct
Tensors.

- Data `[2, 3, 4]`, indices `[5, 2]`, `B=0`, and `K=2` require updates `[5, 4]`; the result is
  `[2, 3, 4]`.
- Data `[2, 3, 4]`, indices `[2, 5, 1]`, `B=1`, and `K=1` require updates `[2, 5, 4]`; the result
  is `[2, 3, 4]`.
- Data `[2, 3]`, indices `[2]`, `B=0`, and `K=2` require canonical rank-zero scalar updates `[]`;
  the result is `[2, 3]`.

Each reduction describes how a base value and all updates addressed to one target combine:

- `NONE` replaces the base value and requires unique target tuples. Duplicate tuples are invalid,
  but detecting them requires index values and is not performed by these semantic values.
- `ADD` combines them by addition.
- `MUL` combines them by multiplication.
- `MAX` combines them by maximum.
- `MIN` combines them by minimum.

The following is a conceptual composition example, not a public Tensor expression:

```java
ScatterNdAttrs attrs = new ScatterNdAttrs(1, ScatterReduction.ADD);
Operation scatterNd = new Operation(ScatterNdKind.SCATTER_ND, attrs);
```

The `Operation` retains the exact kind and attributes references. Its signature enforces
`ScatterNdAttrs` and declares three inputs and one output. `ScatterNdAttrs` rejects a negative batch count before checking
the reduction, and `null` never means `NONE`. Zero, positive values, and `Integer.MAX_VALUE` are
structurally valid batch counts because the attributes contain no input ranks.

Tuple depth remains in the final indices Dimension because it varies by operation occurrence; it
is not duplicated in the attributes. The current public `Tensor.scatterNd` overloads separately
own index/update data-type, rank, shared-batch-prefix, tuple-depth, updates-Shape, result metadata,
and provenance validation. The semantic values themselves read no values, check no bounds or
duplicates, define no gradients or numerical algorithm, capture no graph, report no backend
support, and execute no work. Scatter-ND differs from Gather-ND, which reads selected data, and
from axis scatter, whose indices address one selected axis rather than multi-axis tuples.

### Window-transform semantic kinds and attributes

The public enum
`io.github.pho001.synaptik.model.operation.layout.WindowTransformKind` implements `OperationKind`
with exactly `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, and `FOLD2D`, in that order. These
backend-independent meanings materialize sliding windows or scatter-add window contributions;
they do not construct Tensors, calculate result Shapes, or execute values.

| Kind | Semantic meaning | Required attributes |
|---|---|---|
| `UNFOLD_AXIS` | Materialize no-padding, no-dilation windows along one normalized general axis; replace that extent with window positions and append window size as the final axis. | `UnfoldAxisAttrs` |
| `FOLD_AXIS` | Compiler-only semantic: interpret the eventual input's final dimension as window size, remove it, and scatter-add windows along one normalized target axis with an explicit restored extent. | `FoldAxisAttrs` |
| `UNFOLD2D` | Convert one conceptual rank-four NCHW image tensor to canonical rank-three im2col columns. | `Window2dAttrs` |
| `FOLD2D` | Accumulate canonical rank-three columns through col2im into one explicit rank-four NCHW result Shape. | `Fold2dAttrs` |

NCHW orders axes as batch, channel, height, and width. Im2col places sampled image windows into
columns; col2im scatters those column entries back to image coordinates and adds entries that
target the same coordinate. These names explain the `UNFOLD2D` and `FOLD2D` meanings; they are
not additional operation kinds.

#### General-axis unfold and fold

##### Goal and inputs

Describe an axis unfold for conceptual input Shape `[2, 5, 3]`, then the matching one-dimensional
scatter-add behavior independently. The unfold uses normalized axis `1`, window size `3`, and
step `1`. The fold uses conceptual window input Shape `[3, 3]`, normalized target axis `0`,
explicit output size `5`, and step `1`.

```java
UnfoldAxisAttrs unfoldAttrs = new UnfoldAxisAttrs(1, 3, 1);
FoldAxisAttrs foldAttrs = new FoldAxisAttrs(0, 5, 1);

Operation axisUnfold = new Operation(
        WindowTransformKind.UNFOLD_AXIS,
        unfoldAttrs);
Operation axisFold = new Operation(
        WindowTransformKind.FOLD_AXIS,
        foldAttrs);
```

##### Result and interpretation

For the unfold, the selected extent `5` has three window starts. Its conceptual result Shape is
`[2, 3, 3, 3]`: axis `1` becomes the three-position window count, the former trailing extent `3`
stays in place, and the window size `3` is appended as the new final axis. For a static selected
extent `D`, current public Shape construction calculates:

```text
windowCount = floor((D - size) / step) + 1
```

It first must prove `size <= D` and use checked `long` arithmetic. `UnfoldAxisAttrs` implements
neither check nor calculation. General-axis unfold has no image, padding, or dilation assumption,
and its meaning is materialized windows rather than a promise that a storage view exists.

For the fold, suppose the conceptual `[3, 3]` windows contain
`[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`. Window element `offset` from window `windowIndex` targets
`windowIndex * step + offset`. Scatter-add therefore gives conceptual Shape `[5]` with values
`[1, 6, 15, 14, 9]`. Overlaps sum, and valid target positions receiving no contribution remain
zero. `outputSize` is explicit because window count, final-dimension window size, and step do not
identify trailing positions that the windows did not cover.

The input's final dimension supplies `FOLD_AXIS` window size; `FoldAxisAttrs` deliberately does
not duplicate it. No public Tensor expression constructs this semantic. Task 0023 owns its first
compiler-generated construction for backward graphs, including operand compatibility and gradient
use. Retaining the model value now does not implement autograd.

##### Validation and ownership

Both records store an already normalized non-negative `axis`. Public `Tensor.unfold` accepts a
negative axis while it has the source rank available to normalize and bounds-check the request.
No corresponding public fold-axis normalization boundary exists. `UnfoldAxisAttrs` requires
positive `size` and `step`. `FoldAxisAttrs` accepts non-negative
`outputSize`, including zero, and requires positive `step`. Both check components in declaration
order and retain every valid primitive unchanged, including their maximum values. They contain no
rank or Shape, so construction cannot prove axis bounds, size fit, input-window compatibility, or
arithmetic representability.

#### NCHW im2col and overlap-summing col2im

##### Goal and inputs

Describe `UNFOLD2D` for conceptual NCHW Shape `[1, 1, 3, 3]` with a 2-by-2 kernel, unit stride,
zero symmetric padding, unit dilation, and floor mode, then pair the same window geometry with an
explicit fold output Shape.

```java
Window2dAttrs window = new Window2dAttrs(
        2, 2,
        1, 1,
        0, 0,
        1, 1,
        false);
Fold2dAttrs fold = new Fold2dAttrs(Shape.of(1, 1, 3, 3), window);

Operation imageUnfold = new Operation(
        WindowTransformKind.UNFOLD2D,
        window);
Operation imageFold = new Operation(
        WindowTransformKind.FOLD2D,
        fold);
```

##### Geometry and result

Stride is the positive distance between consecutive window starts. Dilation is the positive
spacing between adjacent kernel samples. Symmetric padding adds the same non-negative logical
width on both sides of a spatial dimension; `UNFOLD2D` treats sampled positions outside the
source as conceptual zeros. Effective kernel is the spatial span covered after dilation. For
height and width independently, current public static-Shape construction uses:

```text
effectiveKernel = dilation * (kernel - 1) + 1
numerator       = input + 2 * padding - effectiveKernel
output          = floor(numerator / stride) + 1       when ceilMode is false
output          = ceil(numerator / stride) + 1        when ceilMode is true
```

Floor mode rounds the quotient down; ceil mode rounds it up. Current public expression
construction performs these calculations with checked `long` arithmetic and requires the
effective kernel to fit the padded dimension. `Window2dAttrs` stores the exact geometry and
rounding flag but evaluates no formula.

The example has output height and width `2`. Canonical im2col Shape is
`[N, C * kernelHeight * kernelWidth, outputHeight * outputWidth]`, so the conceptual result is
`[1, 4, 4]`. Folding compatible columns into explicit Shape `[1, 1, 3, 3]` scatter-adds repeated
coordinates: the center receives four contributions and each corner receives one. Col2im does not
divide by overlap count, and uncovered output positions remain zero.

##### Structural boundary

`Window2dAttrs` requires positive kernel, stride, and dilation dimensions and non-negative
padding dimensions, validating them in component order. It retains valid `long` values and
`ceilMode` unchanged without arithmetic. `Fold2dAttrs` null-checks `outputShape` before `window`
and retains both exact immutable references. The structural record accepts every current Shape
category, including scalar, non-rank-four, zero-extent, and dynamic Shapes, because it has no input
columns to compare. Current public `fold2d` construction establishes the rank-four NCHW/static
compatibility boundary before constructing a fold expression.

Family signatures enforce all four exact attributes pairings and declare one input and one output.
The semantic values themselves define no Tensor construction,
result-data-type rules, resolved layout, storage, provenance, gradients, graph/compiler behavior,
planning, prepare, runtime, backend or ONNX behavior, materialization, or execution. The current
public Tensor methods add only the local descriptor and provenance behavior documented below.

### Unary elementwise semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind` implements
`OperationKind` with exactly these constants in declaration order:

| Kind | Elementwise meaning |
|---|---|
| `ABS` | Absolute magnitude of the input value. |
| `NEG` | Additive inverse of the input value. |
| `RECIPROCAL` | Multiplicative reciprocal of the input value. |
| `LOG` | Natural logarithm of the input value. |
| `LOG1P` | Natural logarithm of one plus the input value. |
| `EXP` | Natural exponential of the input value. |
| `EXPM1` | Natural exponential of the input value minus one. |
| `ERF` | Gaussian error function of the input value. |
| `SQRT` | Principal square root of the input value. |
| `RSQRT` | Reciprocal of the principal square root of the input value. |
| `FLOOR` | Greatest integer-valued result not greater than the input value. |
| `CEIL` | Least integer-valued result not less than the input value. |
| `SIGN` | Negative, zero, or positive sign classification represented numerically. |
| `RELU` | Rectified linear unit of the input value. |
| `SIGMOID` | Logistic sigmoid of the input value. |
| `TANH` | Hyperbolic tangent of the input value. |
| `GELU` | Exact Gaussian error linear unit, `x * Phi(x)`. |
| `GELU_TANH_APPROXIMATION` | Fixed conventional tanh GELU approximation with coefficient `0.044715`. |
| `SILU` | Sigmoid linear unit, `x * sigmoid(x)`. |

All nineteen kinds have one logical input and no intrinsic parameters. Their shared signature
declares exact `NoOperationAttrs`, one input, and one output; canonical composition remains
explicit:

```java
Operation exponential = new Operation(
        UnaryElementwiseKind.EXP,
        NoOperationAttrs.INSTANCE);
```

The enum does not store the input, infer a result descriptor, or create provenance. The current
public unary Tensor methods own those expression-construction rules. `RSQRT`, `LOG1P`, `EXPM1`,
`GELU`, `GELU_TANH_APPROXIMATION`, and `SILU` remain first-class operations rather than
decompositions. The two GELU kinds select distinct exact functions; neither carries configurable
attributes. Together with `EXP` and `TANH`, these are portable mathematical requests: no kind
selects an algorithm, bitwise result, fixed accuracy bound, gradient rule, or backend route. The
selected formulas and special-value semantics are documented in [Unary numeric
transforms and floating classifications](#unary-numeric-transforms-and-floating-classifications).
Inherited enum names are diagnostic text, not serialization or dispatch keys, and equally named
kinds from another family remain different typed values.

### Floating-classification semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind`
implements `OperationKind` with exactly `IS_FINITE`, `IS_NAN`, and `IS_INF`, in that order. Each
kind has exact `NoOperationAttrs`, one input, and one output. The semantic family is separate from
`UnaryElementwiseKind` because its public construction produces fixed non-differentiable BOOL
metadata instead of preserving a floating result type.

The kind values name elementwise classifications only. They do not retain an operand descriptor,
infer BOOL metadata, inspect a NaN encoding or other stored value, create provenance, define
gradients, execute classification, or report backend support. The public Tensor methods own
floating-only input validation, exact Shape retention, unresolved layout, false gradient
eligibility, and one-input provenance. “Classification” here is graph-visible result semantics;
it is unrelated to tracing diagnostics.

### Scalar arithmetic and clamp semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind` implements
`OperationKind` with exactly these constants in declaration order:

| Kind | Elementwise meaning | Required attributes |
|---|---|---|
| `ADD` | Input value plus a scalar addend. | `ScalarValueAttrs` |
| `SUB` | Input value minus a scalar subtrahend. | `ScalarValueAttrs` |
| `MUL` | Input value multiplied by a scalar multiplier. | `ScalarValueAttrs` |
| `DIV` | Input value divided by a scalar denominator. | `ScalarValueAttrs` |
| `MIN` | Pairwise minimum of the input and scalar candidate. | `ScalarValueAttrs` |
| `MAX` | Pairwise maximum of the input and scalar candidate. | `ScalarValueAttrs` |
| `POW` | Input value as the base raised to a scalar exponent. | `ScalarValueAttrs` |
| `CLAMP` | Input value constrained to inclusive lower and upper scalar bounds. | `ClampRangeAttrs` |

The scalar parameter or bounds are operation attributes, not additional Tensor inputs. The family
signatures enforce the table: `CLAMP` accepts exactly `ClampRangeAttrs`, while the other seven kinds
accept exactly `ScalarValueAttrs`; every variant declares one input and one output.

`ScalarValueAttrs` retains one non-null `ScalarValue` by exact reference without conversion,
normalization, or input compatibility checks. `ClampRangeAttrs` requires two non-null values with
the same non-BOOL data type and compares that represented primitive type without a binary64
intermediate. It accepts equal bounds, either ordering of signed zeros, ordered infinities, and
any floating range containing a NaN endpoint. An inverted range fails with
`IllegalArgumentException` and the message
`minValue must be less than or equal to maxValue`.

The following example creates semantic descriptors directly; it does not call the matching public
Tensor expression methods or execute an operation:

```java
Operation scaled = new Operation(
        ScalarElementwiseKind.MUL,
        new ScalarValueAttrs(ScalarValue.float32(-0.0f)));

Operation clipped = new Operation(
        ScalarElementwiseKind.CLAMP,
        new ClampRangeAttrs(ScalarValue.float32(0.0f), ScalarValue.float32(1.0f)));
```

Both attribute records compose `ScalarValue` equality, which includes the exact data type and raw
bits. Positive and negative floating zero are unequal, and distinct NaN payloads remain unequal.
An `Operation` has no operand descriptor, so direct construction does not prove that these FLOAT32
attributes will eventually be attached to a FLOAT32 input.

Scalar MIN/MAX use the same NaN propagation, normal infinity ordering, and opposite-zero choices
as the binary family. `CLAMP(input, lower, upper)` has the value meaning
`minimum(maximum(input, lower), upper)` but remains one semantic occurrence. The one-bound public
methods are conveniences over scalar MAX and MIN rather than separate kinds.

The kinds and attributes alone do not
infer result descriptors, define scalar conversion, establish numerical or special-value execution
behavior, create provenance, define gradients, report backend support, select kernels, or execute
computation. Their enum names and generated record text are diagnostics, not serialization or
dispatch contracts. The current public scalar Tensor methods separately own floating-input
validation, descriptor derivation, exact attribute composition, and one-input provenance.

### Test-local conceptual example

The following types are examples local to a test or explanation. They are not production operation kinds or promises about a future operation-family API.

```java
private enum SampleKind implements OperationKind {
    SAMPLE;

    private static final List<OperationSignature> SIGNATURES = List.of(
            OperationSignature.fixed(SampleAttrs.class, 1, 1));

    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
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

The `SampleKind` declaration shows how an enum inherits `Enum.name()` and owns one stable immutable
signature list. That signature accepts exactly `SampleAttrs`, one input, and one output. The
`SampleAttrs` record shows the intended structural value style: two records with the same `axis`
and `keepDimensions` values compare equal. `NoOperationAttrs.INSTANCE` remains the canonical value
for a kind whose signature explicitly accepts no parameters; it is not accepted by this sample
kind.

`new Operation(kind, attrs)` stores those exact non-null objects without copying or normalization. The second construction uses equal component values, so `operation.equals(equalOperation)` is `true` and their hash codes are equal. The accessors return the original objects: `operation.kind() == kind` and `operation.attrs() == attrs` are both `true`. At the end of the example, `kind.name()` is `"SAMPLE"`, `attrs` contains `axis = 2` and `keepDimensions = true`, and `emptyAttrs` is the singleton `NoOperationAttrs.INSTANCE` available for a parameterless sample kind.

The example demonstrates the implemented descriptor's construction, ownership, exact attributes
compatibility, and record value semantics for an attributes-bearing test value. It does not create
a graph node, infer a result shape or data type, report backend support,
select a kernel, or execute computation. Production `BinaryArithmeticKind`,
`BinaryComparisonKind`, `BooleanLogicalKind`, `WhereSelectionKind`, `UnaryElementwiseKind`, and
`FloatingClassificationKind` provide parameterless families, and
`ScalarElementwiseKind` provides a parameterized family with `ScalarValueAttrs` and
`ClampRangeAttrs`; `CastKind` provides the parameterized cast family with `CastAttrs`. Additional
kind families, compiler behavior, and executable support remain later work in their owning layers.
Binary arithmetic, binary comparison, unary elementwise, floating-classification, and scalar
elementwise semantics have current public Tensor expression methods, as do boolean logical and
conditional-selection semantics. Cast semantics also have current public Tensor expression
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
After those structural checks, the node validates both final list sizes against the operation's
selected signature. Empty inputs or multiple outputs therefore remain available to a kind that
declares them, while a unary signature rejects two inputs. The lists have value semantics: callers
rely on their contents and order, not container identity.

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
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public final class GraphElementExample {
    private enum SampleKind implements OperationKind {
        IDENTITY;

        private static final List<OperationSignature> SIGNATURES = List.of(
                OperationSignature.fixed(NoOperationAttrs.class, 1, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
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
closure and local operation occurrence cardinality. It does not validate descriptor compatibility, prove that the binding
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
The selected operation signature validates local input and output counts; descriptor agreement,
compiler transformations, and execution remain outside the model container.

## Planned contracts

The following contracts appear in the architecture and planning documents but are not implemented:

- random Operations and typed tensor access or export;
- native, mapped, runtime, and backend allocation with deterministic resource ownership;
- expression families beyond binary arithmetic, binary comparison, boolean logical, conditional
  selection, cast, unary elementwise, floating-classification, scalar elementwise, the current
  value- and
  index-producing aggregate operations, cumulative-sum scans, softmax normalization, and
  contiguous, reshape, expand, permute, expand-dimensions, squeeze, slice, pad, tile, concat,
  stack, repeated-select unstack, scalar select, Gather, Gather Elements, Gather-ND Tensor
  construction, Scatter Elements construction, Scatter-ND construction, unfold, unfold2d, and
  fold2d requests; plus gradient
  and trainable state and publication behavior;
- operation-kind families beyond the current binary arithmetic, binary comparison, boolean
  logical, unary elementwise, floating-classification, scalar elementwise, conditional selection,
  cast, aggregate
  reduction, cumulative-sum scan, softmax normalization, contiguous-request, reshape/expand,
  axis-transform, slice, pad, tile, tensor-composition, scalar-select, axis-gather, one-hot,
  gather-ND,
  axis-scatter, scatter-ND, and window-transform semantics, plus
  family-specific attribute values beyond those documented above;
- compiler entry points and transformations, compiler-owned `PublicationPlan` and
  `CompileArtifacts`, and the engine `CompiledGraph` facade; and
- planning, prepare, runtime, publication execution, and backend execution.

`OperationKind`, `OperationAttrs`, `NoOperationAttrs`, `Operation`, `BinaryArithmeticKind`,
`BinaryComparisonKind`, `BooleanLogicalKind`, `WhereSelectionKind`, `UnaryElementwiseKind`,
`FloatingClassificationKind`, `ScalarElementwiseKind`, `ScalarValueAttrs`, `ClampRangeAttrs`,
`CastKind`, `CastAttrs`,
`AggregateReductionKind`, `AxisReductionAttrs`, `MultiAxisReductionAttrs`,
`StatisticalReductionAttrs`, `ArgExtremaTiePolicy`, `ArgExtremaAttrs`,
`MaskedReductionAttrs`, `CumulativeSumKind`, `CumulativeSumAttrs`, `SoftmaxKind`, `SoftmaxAttrs`,
`ContiguousKind`, `ShapeTransformKind`, `TargetShapeAttrs`, `AxisTransformKind`,
`PermutationAttrs`, `AxisTransformAttrs`, `SliceKind`, `SliceAttrs`, `PadKind`, `PadAttrs`,
`TileKind`, `TileAttrs`, `TensorCompositionKind`, `CompositionAxisAttrs`,
`SelectKind`, `SelectAttrs`, `AxisGatherKind`, `IndexAxisAttrs`, `OneHotKind`, `OneHotAttrs`,
`GatherNdKind`, `GatherNdAttrs`, `AxisScatterKind`, `ScatterReduction`, `ScatterElementsAttrs`,
`ScatterNdKind`, `ScatterNdAttrs`,
`WindowTransformKind`, `UnfoldAxisAttrs`, `FoldAxisAttrs`,
`Window2dAttrs`, `Fold2dAttrs`, `GraphValue`, `CompiledNode`,
`GraphPhase`, `CompiledGraphModel`, and
`PublicationBinding`, plus `TensorProducer` and `TensorProvenance`, are current Java API contracts.
The producer/provenance pair can represent shared multi-output occurrences, although no current
production operation or public result API creates one. Binary arithmetic, binary comparison,
boolean logical, unary
elementwise, scalar elementwise, conditional selection, and cast semantics are the current
production concrete kind families with matching public Tensor expression construction. Aggregate
reduction is also a current production semantic family; `sum`, `mean`, `prod`, reduction `min`,
reduction `max`, boolean `all`, boolean `any`, and axis-only `argMin`/`argMax` have matching public Tensor
expression methods, including the masked `sum(axis, mask)` and `mean(axis, mask)` forms. The graph
records can compose these, current cumulative-sum expressions, or test-local semantics, but they
do not provide a compiler entry point or executable support. Softmax and log-softmax semantic
values and their public Tensor expression construction are current. The contiguous semantic value
and public Tensor expression construction are also current, including static resolved and dynamic
unresolved result-layout rules. Reshape and expand semantic values and both public Tensor
expression families are current, with their distinct count-preserving and directional-
broadcasting validation and layout rules.
MATMUL semantics and public `Tensor.matmul` expression construction are current for rank-one and
higher floating or signed-integral operands. Construction promotes within one numeric category,
derives exact vector/matrix/broadcast-batch Shape metadata, defers only the documented unresolved
obligations, and records fresh ordered two-input provenance. It performs no multiplication,
gradient construction, compiler capture, backend selection, or execution.
Axis permutation, singleton-axis insertion, and selected singleton-axis removal semantic values
are current. Public permutation, rank-two transpose, expand-dimensions, and squeeze expression
construction is also current. Signed-step slice semantics and public general/single-axis slice and
flip expression construction are current, including directional static selected-dimension
normalization, canonical zero-extent state, positive-only resolved view geometry, and
negative-step unresolved layout.
Constant-padding and complete-pattern tiling semantic kinds and immutable normalized attributes
are current, as are public pad/tile Tensor request validation, checked Shape derivation, unresolved
result-layout policy, exact metadata retention, and one-input provenance. Padding-constant
conversion, gradients, compiler behavior, materialization, backend/ONNX behavior, and execution
remain planned.
Tensor-composition semantic kinds and public concat/stack construction are current. Public unstack
is ordered repeated scalar select after upfront count validation. Its immutable result list uses
independent one-output producers, provenance index zero, and SELECT's conditional layout behavior.
The general producer/output-index representation remains current. Compiler capture or decomposition, gradients,
materialization, lowering, ONNX mapping, and execution remain planned.
Scalar-select semantics and public `Tensor.select` expression construction are current. The public
method normalizes a raw axis, normalizes and bounds-checks an index when the selected extent is
static, accepts a non-negative index on a dynamic selected extent with its upper bound deferred,
removes the selected axis, derives conditional resolved logical-view geometry, and records fresh
one-input provenance. Value access, a physical alias guarantee, gradients, compiler capture and
canonicalization, materialization, backend lowering, and execution remain planned.
Gather and Gather Elements semantics and public Tensor construction are current. `GATHER` and
`GATHER_ELEMENTS` preserve distinct ordered `[data, indices]` meanings and Shape relationships. Expression
construction validates `INT32`/`INT64` index type, normalizes the data axis, derives and validates
family-specific Shapes, preserves data metadata, leaves layout unresolved, and records fresh
two-input provenance. Index-value access
and bounds checks, gradients, compiler behavior, lowering, and execution remain planned or
separately owned.
Gather-ND semantics and public Tensor construction are current. `GATHER_ND` uses ordered
`[data, indices]` roles, takes tuple depth from the final indices Dimension, and pairs with
`GatherNdAttrs(batchDimensions)`. The public methods validate exact index type, both ranks,
structurally equal batch prefixes, and static positive tuple depth before deriving an
indices-prefix-plus-data-suffix Shape and recording exact provenance. Results retain data type and
gradient eligibility, leave layout unresolved, and remain unlabeled and storage-free. Index-value
bounds, gradients, compiler behavior, lowering, and execution remain separately owned.
Scatter Elements semantics and public Tensor construction are current. `SCATTER_ELEMENTS` uses
ordered `[data, indices, updates]`, preserves the exact data Shape in a new functional result, and
uses `ScatterElementsAttrs` with explicit
`NONE`/`ADD`/`MUL`/`MAX`/`MIN` reduction meaning. Public construction validates exact integral
index type, matching data/update type, reduction eligibility, raw axis, and same-rank Shapes.
Results retain exact data Shape/type, combine
data/update gradient eligibility, leave layout unresolved, and record exact ordered provenance.
`NONE` requires unique targets, but duplicate detection, index bounds, value writes/reductions,
gradient rules, compiler behavior, lowering, and execution remain later responsibilities.
Scatter-ND semantics and public Tensor construction are current.
`SCATTER_ND` uses ordered `[data, indices, updates]` roles, takes tuple depth from the final
indices Dimension, and pairs with `ScatterNdAttrs(batchDimensions, reduction)`. Updates follow the
exact Gather-ND result-Shape formula, and the functional result has the exact data Shape without
in-place mutation. Public construction validates exact index and update types, reduction
eligibility, ranks, shared batch prefix, static positive tuple depth, and the exact updates Shape.
It returns a fresh unresolved-layout, unlabeled, storage-free result with exact data Shape/type,
data/update eligibility OR, and exact ordered provenance. Index values and bounds, duplicate-target
detection, value writes or reductions, gradients, compiler behavior, lowering, backend behavior,
and execution remain planned or separately owned.
Window-transform semantic kinds and immutable normalized attributes are current. Public `unfold`,
`unfold2d`, and `fold2d` expression construction remains current and unchanged, with the same
type/static-Shape compatibility, checked Shape calculation, unresolved result layout, preserved
eligibility, exact attributes, and one-input provenance. No public `foldAxis` method or helper
construction path remains. `FOLD_AXIS` and `FoldAxisAttrs` are compiler-only model semantics;
task 0023 owns their first compiler-generated construction. Gradient construction, compiler
capture and canonicalization, materialization, lowering, backend/ONNX behavior, and execution
remain planned.

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
- `Tensor.slice(long[], long[], int[], long[])` requires four non-null equal-length arrays and
  clones them before element inspection. It normalizes each axis and negative bound once, clamps
  bounds according to step direction, rejects repeated normalized axes and zero steps, and
  preserves rank plus exact unselected Dimension references. Empty arrays and zero-extent results
  are valid; normalized attributes store start plus selected length, not an end sentinel.
- A non-empty all-positive slice of resolved dense, offset, strided, or broadcast geometry derives
  a new checked start-adjusted offset and step-multiplied original strides in a view-marked
  descriptor. Unresolved input, empty results, and every request containing a negative step remain
  unresolved. Every success preserves exact type and eligibility, records normalized `SliceAttrs`
  and `[input]` provenance at output index zero, and is fresh, unlabeled, and storage-free.
  `Tensor.sliceAxis` has step-one and explicit-step forms; `Tensor.flip` creates one negative-step
  SLICE occurrence for its explicit axes, with empty axes meaning identity. None promises a
  physical alias, copy, canonicalization, gradient, materialization, or execution.
- `Tensor.select(int, long)` normalizes one existing axis. A static selected extent normalizes one
  negative index and bounds-checks every resulting coordinate; a dynamic selected extent retains
  a non-negative index with its upper bound deferred and rejects a negative index. The result
  removes exactly that Dimension, preserves every unaffected exact reference, and maps rank one
  to the canonical scalar Shape.
- Resolved input geometry with a non-empty select result produces one new view descriptor after
  checked offset advancement and selected-stride removal. Unresolved input and empty results stay
  unresolved. Every success preserves exact type and eligibility, records normalized
  `SelectAttrs` and `[input]` provenance, and remains fresh, unlabeled, and storage-free without a
  physical alias, gradient, compiler, materialization, backend, or execution guarantee.
- `Tensor.pad(long[], long[], double)` requires two non-null rank-sized arrays, clones them before
  element inspection, rejects negative widths, and performs canonical checked static or symbolic
  addition in before-then-after order. Static zero extents and empty scalar arrays are valid; zero
  widths preserve the exact input Dimension reference. The raw `double` constant is retained
  without conversion.
- `Tensor.tile(long...)` requires one non-null rank-sized array, clones it before element
  inspection, rejects non-positive repeats, and performs canonical checked static or symbolic
  multiplication. Static zero extents and empty scalar arrays are valid; repeat one preserves the
  exact input Dimension reference. Both operations accept every data type, preserve exact type and
  gradient eligibility, always leave result layout unresolved, record normalized attributes and
  `[input]` provenance, and remain fresh, unlabeled, and storage-free without binding or
  evaluating formulas, performing materialization, or executing work.
- `Tensor.unfold` requires rank at least one, a static selected extent, positive size and step,
  and a fitting window. It normalizes the raw source axis against input rank, derives a checked
  window count, appends size as the final dimension, and accepts every current data type.
- `Tensor.unfold2d` requires floating rank-four NCHW input with static channel/height/width, while
  `Tensor.fold2d` requires floating rank-three canonical columns compatible with an explicit
  rank-four NCHW output whose channel/height/width are static and whose batch Dimension matches.
  Both use checked effective-kernel, padded-input, floor/ceil position, channel-window, and total-
  window arithmetic. Every window-transform result preserves type and eligibility, leaves layout
  unresolved, records exact `[input]` provenance, and remains fresh, unlabeled, and storage-free.
- `FOLD_AXIS` and `FoldAxisAttrs` retain their one-input/one-output compiler-only semantic
  contract, but no public Tensor or helper method constructs them. Task 0023 owns first compiler
  generation and operand compatibility.
- Current value objects are immutable and defensively copy caller-owned arrays where applicable.
- Operation-kind implementations must return a non-null, non-blank name and a stable immutable
  non-empty signature list. Operation-attribute implementations must preserve immutable value
  semantics; the marker interface cannot enforce immutability at runtime.
- `Operation` rejects a null kind or attributes value and any incompatible exact attributes class,
  retains valid references unchanged, and exposes the selected family-owned signature without
  storing it as another record component.
- `IndexAxisAttrs` rejects a negative normalized axis with `IllegalArgumentException` and exact
  message `axis must be non-negative: <axis>`. It retains every non-negative `int`, including
  `Integer.MAX_VALUE`, without proving that the axis exists for an eventual data input.
- `ScatterElementsAttrs` checks its normalized axis first with the same exact negative-axis
  failure, then requires a non-null `ScatterReduction` with `NullPointerException("reduction")`.
  It retains every non-negative axis and exact reduction unchanged without validating an input
  rank, Shape, data type, index bound, or duplicate target. `NONE` duplicates remain invalid but
  require later value-aware detection.
- `CompositionAxisAttrs` rejects a negative normalized axis with `IllegalArgumentException` and
  exact message `axis must be non-negative: <axis>`. It retains every non-negative `int`, including
  `Integer.MAX_VALUE`, without proving an input or insertion bound.
- `PadAttrs` null-checks `before` and `after` in component order, rejects unequal sizes, then
  validates paired entries in ascending index order: before null, after null, before negative,
  after negative. Only complete valid input is snapshotted in component order. Empty lists,
  `Long.MAX_VALUE`, signed zero, NaN, and infinities are structurally valid; no rank, Shape,
  `DataType`, or result behavior is inferred.
- `TileAttrs` null-checks `repeats`, then rejects the first null, zero, or negative entry in
  ascending order before taking one immutable snapshot. Empty lists and `Long.MAX_VALUE` are
  structurally valid; no input rank or result-Shape multiplication is performed.
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
  retains the exact non-negative axis and dimension choice. `ArgExtremaAttrs` performs that axis
  check before rejecting a null `ArgExtremaTiePolicy` with `NullPointerException("tiePolicy")`. Ordinary
  full reductions use `NoOperationAttrs.INSTANCE`; no negative all-axis sentinel exists. These
  semantic values do not by themselves validate rank, construct Tensor results, or define
  executable or backend behavior. Public `sum`, `mean`, `prod`, reduction `min`, reduction `max`,
  boolean `all`, boolean `any`, and `argMin`/`argMax` perform the local rank and result construction
  described above. Ordinary integral sum/product use exact-width modular semantics and bounded
  empty identities; integral min/max use signed order and bounded extrema identities. Arg extrema
  accept only floating or integral input, require an explicit policy at their shared helper
  boundary, fix results to non-differentiable INT64, use the documented shared floating/integral
  ordering, and reject a statically empty selected axis.
- `MaskedReductionAttrs` rejects a negative normalized reduction axis with
  `IllegalArgumentException` and exact message `axis must be non-negative: <axis>`. It retains
  every non-negative axis without storing a Shape, mask, mapping, or broadcast plan. The value
  records masked SUM/MEAN semantics only and performs no Shape validation, Tensor construction,
  provenance construction, value selection, counting, or execution. Public masked `sum` and
  `mean` separately require floating input and an exact BOOL mask, normalize one axis, require an
  ordinary right-aligned broadcast whose result equals the input Shape, remove the axis, and
  construct exact `[input, mask]` provenance. Null, type, axis, and broadcast failures occur
  before result identity allocation. Construction performs no storage alignment, numerical work,
  gradient rule, compiler behavior, backend behavior, or execution.
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
  retains the exact `TensorProvenance` reference for the tensor's lifetime without synchronization
  and must select that same exact descriptor reference from its producer position; structural
  descriptor equality alone is insufficient.
  It borrows optional host storage, validates data type, resolved referenced span, and
  attachment-time liveness in that order, and changes only the storage reference through
  synchronized snapshot, replacement, and clearing methods. Failed replacement preserves the
  previous reference, later storage death remains observable, and neither transition changes
  provenance or metadata-only diagnostics.
- `TensorProducer` rejects null components and indexed elements, empty output descriptors, and
  input/output counts outside the selected operation signature. It snapshots ordered inputs and
  output descriptors with exact element references, derives output count from the descriptor
  list, retains no output Tensor, and uses ordinary object identity rather than structural
  equality or an allocated producer ID.
- `TensorProvenance` rejects a null producer and every output index outside
  `[0, producer.outputCount())`. It retains the exact producer and index, then derives the exact
  operation, immutable ordered inputs, and selected descriptor. Record equality uses producer
  identity and the index; neither producer nor provenance establishes graph membership or
  executable support.
- Static `Tensor.concat` and `Tensor.stack` reject a null input array, empty input sequence, or the
  first null copied element before descriptor inspection. Axis validation precedes encounter-order
  type/Shape checks. Both preserve ordered exact inputs, require exact data types, OR gradient
  eligibility, and return fresh unresolved unlabeled storage-free results. CONCAT additionally
  requires equal rank and non-axis Dimensions and encounter-order folds every selected extent
  through canonical checked symbolic addition. STACK requires identical Shapes and inserts a
  static input-count Dimension. Neither binds or evaluates symbolic formulas, promotes,
  broadcasts, reads values/storage, or materializes.
- `Tensor.unstack` normalizes one existing axis, rejects a dynamic selected extent or a static
  extent above `Integer.MAX_VALUE`, then returns one fresh scalar SELECT per coordinate in an
  immutable ordered List. Each output preserves input metadata, uses SELECT's conditional layout,
  and has an independent one-output producer over `[input]` with provenance index zero. A zero
  extent consumes no ID. Mid-request ID exhaustion may consume earlier IDs but returns no partial
  List.
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
- `Tensor.abs`, `neg`, `reciprocal`, `log`, `log1p`, `exp`, `expm1`, `erf`, `sqrt`, `rsqrt`,
  `floor`, `ceil`, `sign`, `relu`, `sigmoid`, and `tanh` accept only floating receiver data types and retain
  the exact data type, shape reference, and gradient-eligibility flag. Each successful call returns
  a fresh unlabeled storage-free Tensor with unresolved layout and exact matching parameterless
  operation plus one-input provenance. Type failures precede ID allocation; identity exhaustion
  follows local descriptor and provenance construction. The methods do not inspect mathematical
  domains, simplify chains, define numerical accuracy, capture graphs, or provide gradient or
  backend behavior. `rsqrt`, `log1p`, and `expm1` are first-class transforms; together with `exp`
  and `tanh`, they select no algorithm, bitwise result, fixed accuracy bound, or backend route.
- `Tensor.isFinite`, `isNaN`, and `isInf` accept only floating receiver data types and construct
  fixed BOOL results with the exact input Shape, unresolved layout, false gradient eligibility,
  and exact matching one-input provenance. They record future value classifications without
  inspecting storage, eagerly producing booleans, defining gradients, capturing a graph, or
  executing work.
- The `ScalarValue` overloads of `Tensor.mul`, `pow`, `clamp`, `clampMin`, and `clampMax` accept
  only floating receivers and require exact receiver/value data-type equality. Their `double`
  overloads adapt only to exact FLOAT64. All forms preserve the input type, shape reference, and
  gradient eligibility. Each
  successful call returns a fresh unlabeled storage-free Tensor with unresolved layout and exact
  matching one-input provenance. `clamp` validates the input type before rejecting only a strictly
  inverted represented range and remains one `CLAMP` operation. Type and range failures precede
  ID allocation; the methods do not convert parameters, simplify expressions, define numerical behavior, capture
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
- Prefix shaping is not a public product capability. Package-private test-source fixtures may
  prepare strict or cyclic carriers and delegate to public flat import, but production source,
  generated production Javadoc, and the TensorFactory surface expose no prefix method.
- `TensorRandoms.randomNormal` requires a caller-owned `RandomGenerator`, a fully static
  Java-array-sized shape, a floating data type, finite mean, and finite numerically non-negative
  standard deviation. It consumes one `nextGaussian()` call per row-major element, transforms in
  binary64 with ordinary multiplication then addition, converts to the exact FLOAT64, FLOAT32, or
  BFLOAT16 carrier, and delegates once to flat import. The source is never retained, substituted,
  synchronized, seeded, reset, split, or closed. Prevalidation and source-carrier allocation
  failures consume no calls or ID; a source exception preserves prior source advancement; blank
  label and exhaustion occur after all calls and destination allocation under the delegated
  flat-import side effects.
- `TensorRandoms.randomUniform` requires a caller-owned `RandomGenerator`, a fully static
  Java-array-sized shape, a floating data type, finite binary64 bounds, and a lower bound strictly
  less than the upper bound. It consumes one bounded `nextDouble(lower, upper)` call per row-major
  element, stores the returned binary64 value directly or narrows it to FLOAT32/BFLOAT16, and
  delegates once to flat import. A conforming source's binary64 result is half-open; narrowing may
  equal the corresponding narrowed upper bound. The source is never retained, substituted,
  synchronized, seeded, reset, split, or closed. Prevalidation and source-carrier allocation
  failures consume no calls or ID; a source exception preserves prior advancement; blank label and
  exhaustion occur after all calls and destination allocation under the delegated flat-import
  side effects.
- The two `TensorRandoms.randomInt` overloads infer `INT32` from primitive `int` bounds and
  `INT64` from primitive `long` bounds. They require a fully static Java-array-sized shape and a
  strict half-open interval, always disable gradients, and consume one matching bounded
  `nextInt(origin, bound)` or `nextLong(origin, bound)` call per row-major element. Each direct
  result is stored without modulo, unbounded sampling, floating arithmetic, or conversion in one
  exact carrier and delegated once to flat import. The same-carrier bound cannot express an
  exclusive value above the carrier maximum, so no full-domain form is provided. Prevalidation
  and source-carrier allocation failures consume no calls or ID; a source exception preserves
  prior calls; blank-label failure and exhaustion occur after all calls and destination
  allocation without rollback.
- `TensorRandoms.randomBernoulli` requires a caller-owned `RandomGenerator`, a fully static
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
- `CompiledNode` snapshots ordered lists, permits repeated inputs, requires non-empty
  within-node-unique outputs, and validates final input/output counts against the selected operation
  signature. It does not perform graph-wide or operand-aware descriptor validation.
- `CompiledGraphModel` snapshots all collections and validates graph-wide structural closure,
  topological order, declared boundaries, producer rules, and exact node-phase coverage. It
  accepts zero-input nodes, repeated node inputs, unused inputs, and zero-node pass-through graphs.
- `PublicationBinding` rejects null identities and remains a standalone association without
  owning-graph validation, policy, storage, backend, or runtime behavior.
- None of the current types owns device storage, runtime residency, or backend selection. The
  current `Tensor` also owns no storage lifetime, graph-local identity, compiler graph capture, or
  execution state; its provenance is model origin metadata only.

See generated Javadoc for the exact member-level exception and nullability contract.
