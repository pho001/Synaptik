# [Padding](../../glossary.md#padding), composition, select, [gather](../../glossary.md#gather) and scatter kinds

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Know whether you are configuring an operation or indexing values

Expression helpers call these attribute constructors with fixed widths, axes or reduction choices,
then pass the immutable objects to `Operation`, which retains them by exact reference.
List snapshots protect configuration from later mutation; they do not copy the Tensor data.
Neither constructor receives stored indices, so bounds and duplicate destinations cannot be
checked here. Read [gather's descriptor/value boundary](gather.md) and
[scatter's duplicate rules](scatter.md#scatter-elements-expressions) before using a kind as
evidence that an indexed request can execute.

**[Reader model](../../glossary.md#term-reader-model).** Indexing uses integer [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) to relate input and output positions.
**Gather** reads selected sources; **scatter** sends updates to selected destinations.
The [ND](../../glossary.md#term-nd) forms use coordinate tuples addressing several [axes](../../glossary.md#term-axis). **Functional** scatter produces
a new result rather than mutating its base. **Composition** joins [tensors](../../glossary.md#tensor): concat uses an
existing axis, while stack creates one. Padding adds border positions and [tiling](../../glossary.md#tiling) repeats a
complete pattern. These kind and attribute values describe the requests; the
[gather](gather.md), [scatter](scatter.md), and [composition](composition.md) pages own the
input-aware [shape](../../glossary.md#shape) rules. A [signature](../../glossary.md#term-signature)'s input count counts tensors, not coordinates or [elements](../../glossary.md#term-element).

<a id="pad-and-tile-semantic-kinds-and-normalized-attributes"></a>
### Pad and tile semantic kinds and normalized attributes

The public enums `io.github.pho001.synaptik.model.operation.layout.PadKind` and
`io.github.pho001.synaptik.model.operation.layout.TileKind` implement [`OperationKind`](../../glossary.md#operationkind) with exactly
[`PAD`](../../glossary.md#padding) and `TILE`, respectively. `PAD` means adding [constant](../../glossary.md#term-constant)-filled logical positions before and
after every input [axis](../../glossary.md#term-axis) without changing [rank](../../glossary.md#term-rank). `TILE` means repeating the complete input pattern a
positive number of times along every axis without changing rank. The kinds identify meaning only;
they do not inspect an input or calculate values or extents.

`PadAttrs(before, after, constantValue)` stores two ordered equal-size `List<Long>` width lists and
one raw Java `double`. Entry `i` in each list applies before or after the same [normalized axis](../../glossary.md#normalized-axis)
position. `TileAttrs(repeats)` stores one ordered `List<Long>` whose entry `i` is the number of
complete input-pattern repetitions along normalized axis position `i`.

<a id="constant-padding-example"></a>
#### [Constant](../../glossary.md#term-constant)-[padding](../../glossary.md#padding) example

<a id="goal-and-inputs-43"></a>
##### Goal and inputs

Describe one-dimensional [constant](../../glossary.md#term-constant) [padding](../../glossary.md#padding) for conceptual input `[10, 20]`: add one position
before, two positions after, and fill those logical positions with `-1`.

```java
// Describe border widths; an input-aware call must still check rank and constant type.
PadAttrs attrs = new PadAttrs(List.of(1L), List.of(2L), -1.0d);
Operation padded = new Operation(PadKind.PAD, attrs);
```

<a id="result-and-interpretation-42"></a>
##### Result and interpretation

The request means `[-1, 10, 20, -1, -1]`. `padded.kind()` is exactly `PadKind.PAD`, and
`padded.attrs()` is exactly the supplied `attrs` reference. This is a conceptual semantic result:
construction does not create a [Tensor](../../glossary.md#tensor), populate values, derive a [Shape](../../glossary.md#shape) or [layout](../../glossary.md#layout), convert `-1.0d`
to an input [`DataType`](../../glossary.md#data-type--datatype), or execute [padding](../../glossary.md#padding).

<a id="complete-pattern-tiling-example"></a>
#### Complete-pattern [tiling](../../glossary.md#tiling) example

<a id="goal-and-inputs-44"></a>
##### Goal and inputs

Describe [tiling](../../glossary.md#tiling) conceptual input `[[1, 2], [3, 4]]` twice along [axis](../../glossary.md#term-axis) zero and three times along
axis one.

```java
// Counts repeat the complete pattern, not each individual value consecutively.
TileAttrs attrs = new TileAttrs(List.of(2L, 3L));
Operation tiled = new Operation(TileKind.TILE, attrs);
```

<a id="result-and-interpretation-43"></a>
##### Result and interpretation

The complete row pattern repeats three times and the complete two-row pattern repeats twice:

```text
[[1, 2, 1, 2, 1, 2],
 [3, 4, 3, 4, 3, 4],
 [1, 2, 1, 2, 1, 2],
 [3, 4, 3, 4, 3, 4]]
```

This is complete-pattern [tiling](../../glossary.md#tiling), not [scalar](../../glossary.md#term-scalar)-[element](../../glossary.md#term-element) repeat: it does not transform `[1, 2]` into
`[1, 1, 1, 2, 2, 2]`. `tiled.kind()` is exactly `TileKind.TILE`, and `tiled.attrs()` is exactly
the supplied `attrs` reference. The example states requested logical meaning without claiming
[Tensor](../../glossary.md#tensor) construction, value [materialization](../../glossary.md#materialization), or execution.

<a id="ownership-validation-and-useful-variations-1"></a>
##### [Ownership](../../glossary.md#term-ownership-meanings), [validation](../../glossary.md#term-validation), and useful variations

Both records validate caller-owned lists in ascending index order and copy only after every
[element](../../glossary.md#term-element) passes. The stored [immutable](../../glossary.md#term-immutable) snapshots preserve order and values but not caller list
[identity](../../glossary.md#term-identity-meanings); later caller mutation cannot change them, and accessor mutation fails. Empty pad lists
and an empty repeat list are valid [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../glossary.md#term-scalar) identity [parameters](../../glossary.md#term-parameter). `Long.MAX_VALUE` is also
structurally valid because these records perform no input-[rank](../../glossary.md#term-rank) matching or result-[Shape](../../glossary.md#shape) arithmetic.

`PadAttrs` validation order is exact:

1. Null-check `before`, then `after`, producing `NullPointerException` with the component name.
2. Reject unequal sizes with `IllegalArgumentException("before and after must have matching sizes")`.
3. At each ascending index, null-check `before[i]`, then `after[i]`, using that indexed message.
4. Reject a negative before width with `before[i] must be non-negative: value`, then a negative
   after width with `after[i] must be non-negative: value`.
5. [Snapshot](../../glossary.md#term-snapshot) `before`, then `after` exactly once each.

Every `constantValue` is retained bit-for-bit as the supplied primitive, including positive and
[negative zero](../../glossary.md#nan-infinity-and-signed-zero), [finite](../../glossary.md#term-finite) values, [NaN](../../glossary.md#term-nan) payloads, and either [infinity](../../glossary.md#term-infinity). No finiteness, range, [rounding](../../glossary.md#term-rounding),
[saturation](../../glossary.md#term-saturation), [BOOL](../../glossary.md#data-type--datatype) [normalization](../../glossary.md#term-normalization), or [`DataType`](../../glossary.md#data-type--datatype) compatibility rule is applied. Ordinary Java record
and `double` equality/[hash](../../glossary.md#term-hash) semantics still apply; raw-bit retention is not a custom equality or
[serialization](../../glossary.md#term-serialization) contract.

`TileAttrs` first null-checks `repeats`, then visits entries in ascending order. A null entry fails
with `NullPointerException("repeats[i]")`; a zero or negative entry fails with
`IllegalArgumentException("repeats[i] must be positive: value")`. Only then does construction take
its single immutable snapshot. Result-Shape multiplication and [overflow](../../glossary.md#term-overflow) checks are deferred.

Family [signatures](../../glossary.md#term-signature) enforce [`PAD`](../../glossary.md#padding) with `PadAttrs` and `TILE` with `TileAttrs`, each with one input
and one output. Neither family defines public [Tensor](../../glossary.md#tensor) request syntax, result [inference](../../glossary.md#term-inference), [layout](../../glossary.md#layout) or
storage behavior, [materialization](../../glossary.md#materialization), [provenance](../../glossary.md#provenance), [gradients](../../glossary.md#term-gradient), [compiler](../../glossary.md#term-compiler) or [planning](../../glossary.md#term-planning) behavior, [backend](../../glossary.md#backend)
or [ONNX](../../glossary.md#term-onnx) behavior, or execution.

<a id="tensor-composition-semantic-kinds-and-attributes"></a>
### [Tensor composition](../../glossary.md#tensor-composition) semantic kinds and attributes

`TensorCompositionKind` implements [`OperationKind`](../../glossary.md#operationkind) with exactly `CONCAT` and `STACK`, in
that order. Both pair with `CompositionAxisAttrs` and accept one or more ordered inputs with
exactly one output. CONCAT joins along an existing normalized input [axis](../../glossary.md#term-axis) and preserves [rank](../../glossary.md#term-rank); STACK
joins same-shaped inputs along a newly inserted normalized result axis and increases rank by one.
The semantic types do not validate input rank, type, [Shape](../../glossary.md#shape), or result [metadata](../../glossary.md#term-metadata).

Unstack is not a `TensorCompositionKind` value and has no dedicated attributes. Public
`Tensor.unstack` is the repeated-SELECT convenience described under
[tensor composition expressions](composition.md#tensor-composition-expressions). This keeps each selection a
complete independent [occurrence](../../glossary.md#term-occurrence) and avoids confusing a convenience list with one multi-output
[operation](../../glossary.md#operation).

<a id="scalar-select-semantic-kind-and-attributes"></a>
### [Scalar select](../../glossary.md#scalar-select) semantic kind and attributes

The public enum `io.github.pho001.synaptik.model.operation.index.SelectKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly one [constant](../../glossary.md#term-constant), `SELECT`. It identifies [scalar](../../glossary.md#term-scalar)-index selection: fix
one [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) on one existing source [axis](../../glossary.md#term-axis) and remove that axis from the logical result.
`SelectAttrs(axis, index)` carries the normalized zero-based source-axis position as an `int` and
the normalized zero-based scalar coordinate as a `long`.

<a id="scalar-axis-removal-example"></a>
#### [Scalar](../../glossary.md#term-scalar)-[axis](../../glossary.md#term-axis) removal example

<a id="goal-and-inputs-45"></a>
##### Goal and inputs

Describe selecting [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) `2` on [axis](../../glossary.md#term-axis) `1` from a conceptual source with [Shape](../../glossary.md#shape) `[2, 3, 4]`.

```java
// The coordinate is normalized already; this record cannot prove an input bound.
SelectAttrs attrs = new SelectAttrs(1, 2L);
Operation select = new Operation(SelectKind.SELECT, attrs);
```

<a id="result-and-interpretation-44"></a>
##### Result and interpretation

[Axis](../../glossary.md#term-axis) `1` has [extent](../../glossary.md#term-extent) `3`. Fixing its [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) at index `2` leaves axes `0` and `2`, so the
conceptual result [Shape](../../glossary.md#shape) is `[2, 4]`. `select.kind()` is exactly `SelectKind.SELECT`, and
`select.attrs()` is exactly the supplied `attrs` reference.

The example explains semantic meaning only. Constructing these values does not inspect an input,
derive a result Shape or [layout](../../glossary.md#layout), construct a [Tensor](../../glossary.md#tensor) or [provenance](../../glossary.md#provenance), read a value, or execute work.

<a id="validation-and-boundaries"></a>
##### [Validation](../../glossary.md#term-validation) and boundaries

`SelectAttrs` checks the [axis](../../glossary.md#term-axis) first and rejects a negative value with
`IllegalArgumentException("axis must be non-negative: <axis>")`. Once the axis is valid, it
rejects a negative index with
`IllegalArgumentException("index must be non-negative: <index>")`. Zero, `Integer.MAX_VALUE`,
and `Long.MAX_VALUE` are structurally valid, and both primitive values are retained unchanged.

The attributes store no input [Shape](../../glossary.md#shape), [rank](../../glossary.md#term-rank), or selected-axis [extent](../../glossary.md#term-extent). Structural validity therefore
does not prove that the axis exists or that the index is in bounds. The family [signature](../../glossary.md#term-signature) enforces
`SELECT` with `SelectAttrs` and declares one input and one output.

`SELECT` differs from conditional `WHERE`, which chooses between branch values position by
position, and general [`SLICE`](../../glossary.md#slice), which selects [half-open](../../glossary.md#term-half-open-interval) intervals without removing an axis. It also
differs from [gather](../../glossary.md#gather) operations whose indices are [tensors](../../glossary.md#tensor) rather than one intrinsic [scalar](../../glossary.md#term-scalar)
[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix). Public `Tensor.select` now supplies input-aware axis/index [normalization](../../glossary.md#term-normalization),
validation, result Shape and conditional logical-[view](../../glossary.md#term-view-meanings) [layout](../../glossary.md#layout), and exact one-input [provenance](../../glossary.md#provenance) as
described under [scalar select expressions](gather.md#scalar-select-expressions)<span data-glossary-navigation="true"> ([scalar select glossary](../../glossary.md#scalar-select))</span>. [Compiler](../../glossary.md#term-compiler)
[capture](../../glossary.md#term-capture-meanings), verification and selected floating SELECT cotangents are implemented, as are
qualified [backend](../../glossary.md#backend) [lowering](../../glossary.md#lowering), preparation and execution. No such stage runs inside `SelectAttrs`.

<a id="gather-and-gather-elements-semantic-kinds-and-attributes"></a>
### [Gather](../../glossary.md#gather) and [Gather Elements](../../glossary.md#gather-elements) semantic kinds and attributes

`AxisGatherKind` implements [`OperationKind`](../../glossary.md#operationkind) with exactly `GATHER` and `GATHER_ELEMENTS`, in
that order. Both consume ordered logical inputs `[data, indices]`, pair with
`IndexAxisAttrs(axis)`, and declare exactly two inputs and one output.

- `GATHER` replaces the selected data [axis](../../glossary.md#term-axis) with the complete indices [Shape](../../glossary.md#shape). Data
  `[2, 3, 4]`, axis `1`, and indices `[5, 6]` conceptually produce `[2, 5, 6, 4]`;
  scalar indices produce <span data-glossary-navigation="true"> ([scalar glossary](../../glossary.md#term-scalar))</span>`[2, 4]`.
- `GATHER_ELEMENTS` requires same-rank indices aligned with data away from the selected axis and
  conceptually produces the exact indices Shape. Data `[2, 3, 4]`, axis `1`, and indices
  `[2, 7, 4]` produce `[2, 7, 4]`.

`IndexAxisAttrs` rejects a negative [normalized axis](../../glossary.md#normalized-axis) but contains no data [rank](../../glossary.md#term-rank) or Shape. Public
`Tensor.gather` and `gatherElements` own index-type [validation](../../glossary.md#term-validation), caller-axis [normalization](../../glossary.md#term-normalization),
Shape checks and derivation, result [metadata](../../glossary.md#term-metadata), and exact `[data, indices]` [provenance](../../glossary.md#provenance). Neither
semantic values nor public expression construction inspect index values or bounds, define
[gradients](../../glossary.md#term-gradient), provide [compiler](../../glossary.md#term-compiler) behavior, choose a [backend](../../glossary.md#backend), or execute work.

<a id="one-hot-semantic-kind-and-attributes"></a>
### One-hot semantic kind and attributes

The public enum `io.github.pho001.synaptik.model.operation.index.OneHotKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly `ONE_HOT`. It consumes ordered logical input `[indices]`, produces
one dense [BOOL](../../glossary.md#data-type--datatype) output, requires exact `OneHotAttrs`, and declares exactly one input and one output.
`OneHotAttrs(depth)` retains one positive `long` unchanged. It rejects zero or a negative value
with `depth must be positive: <depth>` and accepts every positive value through
`Long.MAX_VALUE` without claiming that later storage can materialize the [Shape](../../glossary.md#shape).

For an index value `i`, output [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) `j` on the new trailing [axis](../../glossary.md#term-axis) is true exactly when
`i == j`. Eventual execution requires `0 <= i < depth`; an invalid index is not a request for
wrapping, clamping, a default row, or an all-false row. The kind and attributes contain no input
[Tensor](../../glossary.md#tensor), Shape, axis, on/off values, output type, bounds policy, [compiler](../../glossary.md#term-compiler) state, [backend](../../glossary.md#backend) support,
storage, or execution state. Public `Tensor.oneHot` separately owns input-type and depth
[validation](../../glossary.md#term-validation), Shape/[descriptor](../../glossary.md#term-descriptor) construction, and exact one-input [provenance](../../glossary.md#provenance) as described under
[one-hot encoding expressions](gather.md#one-hot-encoding-expressions)<span data-glossary-navigation="true"> ([one-hot encoding glossary](../../glossary.md#one-hot-encoding))</span>.

<a id="gather-nd-semantic-kind-and-attributes"></a>
### [Gather-ND](../../glossary.md#gather-nd) semantic kind and attributes

The public enum `io.github.pho001.synaptik.model.operation.index.GatherNdKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly one [constant](../../glossary.md#term-constant), `GATHER_ND`. It consumes ordered logical inputs
`[data, indices]`: `data` supplies values, while `indices` supplies [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) tuples. The final
indices [Dimension](../../glossary.md#dimension) is the [tuple depth](../../glossary.md#term-tuple-depth) `K`. `GatherNdAttrs(batchDimensions)` stores only the
already normalized, non-negative number `B` of shared leading [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) Dimensions.

For data [rank](../../glossary.md#term-rank) `R` and indices rank `Q`, every tuple indexes data [axes](../../glossary.md#term-axis) `[B, B + K)`. Data axes
`[B + K, R)` form the untouched suffix of the selected value. The conceptual result [Shape](../../glossary.md#shape) is:

```text
indices.shape[0:Q-1] + data.shape[B+K:R]
```

The first term is equivalently `indices.shape[:-1]`: every indices Dimension except the final
tuple-depth Dimension remains in the result.

<a id="tuple-index-shape-examples"></a>
#### Tuple-index [Shape](../../glossary.md#shape) examples

<a id="goal-and-inputs-46"></a>
##### Goal and inputs

Compare zero-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) selection, batched selection, and selection of one complete [scalar](../../glossary.md#term-scalar):

```java
// Shared batch count is configuration; tuple depth still comes from the indices shape.
GatherNdAttrs attrs = new GatherNdAttrs(1);
Operation gatherNd = new Operation(GatherNdKind.GATHER_ND, attrs);
```

- Data `[2, 3, 4]`, indices `[5, 2]`, `B=0`, and `K=2` select two data [axes](../../glossary.md#term-axis) per tuple.
- Data `[2, 3, 4]`, indices `[2, 5, 1]`, `B=1`, and `K=1` share the leading [extent](../../glossary.md#term-extent) `2` and select
  one later data axis per tuple.
- Data `[2, 3]`, indices `[2]`, `B=0`, and `K=2` select every data axis with one tuple.

<a id="results-and-interpretation"></a>
##### Results and interpretation

- The first result is `[5, 4]`: indices prefix `[5]` is followed by untouched data suffix `[4]`.
- The second result is `[2, 5, 4]`: indices prefix `[2, 5]` is followed by suffix `[4]`.
- The third result is [canonical](../../glossary.md#term-canonical-form) [scalar](../../glossary.md#term-scalar) [Shape](../../glossary.md#shape) `[]`, not `[1]`, because both formula terms are
  empty.

The [`Operation`](../../glossary.md#operation) in the code example retains exactly `GatherNdKind.GATHER_ND` and the supplied
`attrs` reference. The current zero-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) `Tensor.gatherNd(indices)` convenience uses
`new GatherNdAttrs(0)` rather than a second kind or a default attributes value.

<a id="validation-and-boundaries-1"></a>
##### [Validation](../../glossary.md#term-validation) and boundaries

`GatherNdAttrs` rejects a negative count with
`IllegalArgumentException("batchDimensions must be non-negative: <batchDimensions>")`. Zero,
positive values, and `Integer.MAX_VALUE` are structurally valid and retained unchanged. The value
contains no input ranks or [Shapes](../../glossary.md#shape), so it cannot prove that `B` fits the inputs, that leading [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)
[Dimensions](../../glossary.md#dimension) match, that the final indices Dimension supplies a valid [tuple depth](../../glossary.md#term-tuple-depth), or that an index
[data type](../../glossary.md#data-type--datatype) is permitted. The family [signature](../../glossary.md#term-signature) enforces `GatherNdAttrs` and declares two inputs and
one output.

Tuple depth remains the final indices Dimension instead of an attribute because it varies with
each [operation](../../glossary.md#operation) [occurrence](../../glossary.md#term-occurrence). The current public [Tensor](../../glossary.md#tensor) boundary validates data and indices ranks,
shared batch Dimensions, tuple depth, index type, and result Shape and records exact [provenance](../../glossary.md#provenance) as
described under [Gather-ND expressions](scatter.md#gather-nd-expressions). The semantic values themselves do
none of that input-aware work. [Gather-ND](../../glossary.md#gather-nd) is distinct from [scalar](../../glossary.md#term-scalar) `SELECT`, which stores one
[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix); [axis](../../glossary.md#term-axis) [gather](../../glossary.md#gather), which indexes one selected axis; and [scatter-ND](../../glossary.md#scatter-nd), which writes or combines
updates. Neither semantic values nor current expression construction define index-value bounds,
a numerical algorithm, [gradients](../../glossary.md#term-gradient), [compiler](../../glossary.md#term-compiler) behavior, [backend](../../glossary.md#backend) behavior, or execution.

<a id="axis-scatter-semantic-kinds-reduction-and-attributes"></a>
### [Axis](../../glossary.md#term-axis)-scatter semantic kinds, reduction, and attributes

`AxisScatterKind` implements [`OperationKind`](../../glossary.md#operationkind) with exactly `SCATTER_ELEMENTS`, then `SCATTER_ADD`.
Both consume ordered logical inputs `[data, indices, updates]` and declare exactly three inputs
and one output. `SCATTER_ELEMENTS` pairs only with `ScatterElementsAttrs(axis, reduction)`;
`SCATTER_ADD` pairs only with `IndexAxisAttrs(axis)`. Cross-family attributes are invalid.

Indices and updates have equal [rank](../../glossary.md#term-rank) and [Shape](../../glossary.md#shape) and match data away from the selected axis. The
functional result has the exact data Shape and does not mutate data. For data `[2, 3, 4]`, axis
`1`, and indices/updates `[2, 5, 4]`, update [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) `[0, 4, 2]` uses its index to select
[target](../../glossary.md#term-target) `[0, index, 2]`.

`ScatterReduction` contains exactly `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`.
`NONE` means replacement and requires unique targets; the arithmetic values combine the base
and all updates for a target. For `SCATTER_ELEMENTS`, every updates coordinate contributes once
to the target obtained by replacing its selected-axis coordinate with the corresponding index.
The base participates once, duplicates remain distinct contributions, and an unaddressed
coordinate retains the exact base representation. Configurable `MUL`, `MIN`, and `MAX` have the
portable, order-independent represented-value contract specified under
[Scatter Elements expressions](scatter.md#scatter-elements-expressions)<span data-glossary-navigation="true"> ([Scatter Elements glossary](../../glossary.md#scatter-elements))</span>; the vocabulary selects no
traversal, [atomic](../../glossary.md#resource-transaction-and-rollback), tree, or [backend](../../glossary.md#backend) algorithm and no [derivative](../../glossary.md#term-derivative) or subgradient policy.

`SCATTER_ADD` instead requires updates with the exact Shape that ordinary [Gather](../../glossary.md#gather) would produce:
data prefix, complete indices Shape, then data suffix. It has intrinsic addition rather than a
stored reduction. Its data-shaped functional result includes the base and all updates for every
target, so duplicate indices accumulate. `IndexAxisAttrs` is shared with axis Gather because the
same normalized data axis defines both Shape relationships.

`ScatterElementsAttrs` validates the normalized non-negative axis before requiring a non-null
reduction. Public `Tensor.scatterElements` owns caller-axis, type, reduction-eligibility, rank,
Shape, [descriptor](../../glossary.md#term-descriptor), and [provenance](../../glossary.md#provenance) [validation](../../glossary.md#term-validation). Index bounds and duplicate targets require values
and remain outside [model](../../glossary.md#term-model) [metadata](../../glossary.md#term-metadata) construction. Public `Tensor.scatterAdd` owns the corresponding
numeric and exact Gather-compatible Shape checks. The semantic values define no [gradients](../../glossary.md#term-gradient),
[compiler](../../glossary.md#term-compiler)-generated adjoints, [materialization](../../glossary.md#materialization), [lowering](../../glossary.md#lowering), backend behavior, or execution.

<a id="scatter-nd-semantic-kind-and-attributes"></a>
### [Scatter-ND](../../glossary.md#scatter-nd) semantic kind and attributes

The public enum `io.github.pho001.synaptik.model.operation.index.ScatterNdKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly one [constant](../../glossary.md#term-constant), `SCATTER_ND`. Functional Scatter-ND starts from base
`data`, uses [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) tuples from `indices` to address result [targets](../../glossary.md#term-target), applies `updates` through
an explicit `ScatterReduction`, and produces a new result with the exact data [Shape](../../glossary.md#shape). It does not
mutate `data` in place. Its ordered logical inputs are `[data, indices, updates]`.

`ScatterNdAttrs(batchDimensions, reduction)` stores the already normalized non-negative count
`B` of shared leading data and indices [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [Dimensions](../../glossary.md#dimension), followed by the explicit non-null
reduction. If data [rank](../../glossary.md#term-rank) is `R`, indices rank is `Q`, and the final indices Dimension has [extent](../../glossary.md#term-extent)
`K`, that extent is the [tuple depth](../../glossary.md#term-tuple-depth). Each tuple indexes data [axes](../../glossary.md#term-axis) `[B, B + K)`, while data axes
`[B + K, R)` form the untouched suffix represented by one update [slice](../../glossary.md#slice). The required updates
Shape is:

```text
indices.shape[0:Q-1] + data.shape[B+K:R]
```

Equivalently, it is `indices.shape[:-1] + data.shape[batchDimensions + K:]`.

<a id="tuple-index-update-shape-examples"></a>
#### Tuple-index update [Shape](../../glossary.md#shape) examples

These examples are conceptual; the current semantic values do not inspect operands or construct
[Tensors](../../glossary.md#tensor).

- Data `[2, 3, 4]`, indices `[5, 2]`, `B=0`, and `K=2` require updates `[5, 4]`; the result is
  `[2, 3, 4]`.
- Data `[2, 3, 4]`, indices `[2, 5, 1]`, `B=1`, and `K=1` require updates `[2, 5, 4]`; the result
  is `[2, 3, 4]`.
- Data `[2, 3]`, indices `[2]`, `B=0`, and `K=2` require [canonical](../../glossary.md#term-canonical-form) [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../glossary.md#term-scalar) updates `[]`;
  the result is `[2, 3]`.

Each reduction describes how a base value and all updates addressed to one [target](../../glossary.md#term-target) combine:

- `NONE` replaces the base value and requires unique target tuples. Duplicate tuples are invalid,
  but detecting them requires index values and is not performed by these semantic values.
- `ADD` combines them by addition.
- `MUL`, `MAX`, and `MIN` use the portable, order-independent represented-value contract specified
  under [Scatter Elements expressions](scatter.md#scatter-elements-expressions)<span data-glossary-navigation="true"> ([Scatter Elements glossary](../../glossary.md#scatter-elements))</span>.

Every tuple position contributes its complete update suffix [slice](../../glossary.md#slice) scalar by scalar. The base at
one concrete result [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) participates exactly once, and every addressed update scalar
participates exactly once. Duplicate tuples remain distinct contributions; an unaddressed
coordinate preserves the exact base representation. These forward semantics select no [derivative](../../glossary.md#term-derivative)
or subgradient policy.

The following is a conceptual composition example, not a public Tensor expression:

```java
// ADD permits duplicate destinations semantically; no index values are read here.
ScatterNdAttrs attrs = new ScatterNdAttrs(1, ScatterReduction.ADD);
Operation scatterNd = new Operation(ScatterNdKind.SCATTER_ND, attrs);
```

The [`Operation`](../../glossary.md#operation) retains the exact kind and attributes references. Its [signature](../../glossary.md#term-signature) enforces
`ScatterNdAttrs` and declares three inputs and one output. `ScatterNdAttrs` rejects a negative [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) count before checking
the reduction, and `null` never means `NONE`. Zero, positive values, and `Integer.MAX_VALUE` are
structurally valid batch counts because the attributes contain no input ranks.

[Tuple depth](../../glossary.md#term-tuple-depth) remains in the final indices [Dimension](../../glossary.md#dimension) because it varies by operation [occurrence](../../glossary.md#term-occurrence); it
is not duplicated in the attributes. The current public `Tensor.scatterNd` overloads separately
own index/update data-type, [rank](../../glossary.md#term-rank), shared-batch-prefix, tuple-depth, updates-Shape, result [metadata](../../glossary.md#term-metadata),
and [provenance](../../glossary.md#provenance) [validation](../../glossary.md#term-validation). The semantic values themselves read no values, check no bounds or
duplicates, define no derivatives or numerical algorithm, [capture](../../glossary.md#term-capture-meanings) no [graph](../../glossary.md#graph), report no [backend](../../glossary.md#backend)
support, and execute no work. [Scatter-ND](../../glossary.md#scatter-nd) differs from [Gather-ND](../../glossary.md#gather-nd), which reads selected data, and
from [axis](../../glossary.md#term-axis) scatter, whose indices address one selected axis rather than multi-axis tuples.


---

[Previous topic](operation-geometry.md) · [Next topic](operation-windows.md)
