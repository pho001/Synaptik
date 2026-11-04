# Padding, composition, select, gather and scatter kinds

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="pad-and-tile-semantic-kinds-and-normalized-attributes"></a>
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

<a id="constant-padding-example"></a>
#### Constant-padding example

<a id="goal-and-inputs-43"></a>
##### Goal and inputs

Describe one-dimensional constant padding for conceptual input `[10, 20]`: add one position
before, two positions after, and fill those logical positions with `-1`.

```java
PadAttrs attrs = new PadAttrs(List.of(1L), List.of(2L), -1.0d);
Operation padded = new Operation(PadKind.PAD, attrs);
```

<a id="result-and-interpretation-42"></a>
##### Result and interpretation

The request means `[-1, 10, 20, -1, -1]`. `padded.kind()` is exactly `PadKind.PAD`, and
`padded.attrs()` is exactly the supplied `attrs` reference. This is a conceptual semantic result:
construction does not create a Tensor, populate values, derive a Shape or layout, convert `-1.0d`
to an input `DataType`, or execute padding.

<a id="complete-pattern-tiling-example"></a>
#### Complete-pattern tiling example

<a id="goal-and-inputs-44"></a>
##### Goal and inputs

Describe tiling conceptual input `[[1, 2], [3, 4]]` twice along axis zero and three times along
axis one.

```java
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

This is complete-pattern tiling, not scalar-element repeat: it does not transform `[1, 2]` into
`[1, 1, 1, 2, 2, 2]`. `tiled.kind()` is exactly `TileKind.TILE`, and `tiled.attrs()` is exactly
the supplied `attrs` reference. The example states requested logical meaning without claiming
Tensor construction, value materialization, or execution.

<a id="ownership-validation-and-useful-variations-1"></a>
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

<a id="tensor-composition-semantic-kinds-and-attributes"></a>
### Tensor composition semantic kinds and attributes

`TensorCompositionKind` implements `OperationKind` with exactly `CONCAT` and `STACK`, in
that order. Both pair with `CompositionAxisAttrs` and accept one or more ordered inputs with
exactly one output. CONCAT joins along an existing normalized input axis and preserves rank; STACK
joins same-shaped inputs along a newly inserted normalized result axis and increases rank by one.
The semantic types do not validate input rank, type, Shape, or result metadata.

Unstack is not a `TensorCompositionKind` value and has no dedicated attributes. Public
`Tensor.unstack` is the repeated-SELECT convenience described under
[tensor composition expressions](composition.md#tensor-composition-expressions). This keeps each selection a
complete independent occurrence and avoids confusing a convenience list with one multi-output
operation.

<a id="scalar-select-semantic-kind-and-attributes"></a>
### Scalar select semantic kind and attributes

The public enum `io.github.pho001.synaptik.model.operation.index.SelectKind` implements
`OperationKind` with exactly one constant, `SELECT`. It identifies scalar-index selection: fix
one coordinate on one existing source axis and remove that axis from the logical result.
`SelectAttrs(axis, index)` carries the normalized zero-based source-axis position as an `int` and
the normalized zero-based scalar coordinate as a `long`.

<a id="scalar-axis-removal-example"></a>
#### Scalar-axis removal example

<a id="goal-and-inputs-45"></a>
##### Goal and inputs

Describe selecting coordinate `2` on axis `1` from a conceptual source with Shape `[2, 3, 4]`.

```java
SelectAttrs attrs = new SelectAttrs(1, 2L);
Operation select = new Operation(SelectKind.SELECT, attrs);
```

<a id="result-and-interpretation-44"></a>
##### Result and interpretation

Axis `1` has extent `3`. Fixing its coordinate at index `2` leaves axes `0` and `2`, so the
conceptual result Shape is `[2, 4]`. `select.kind()` is exactly `SelectKind.SELECT`, and
`select.attrs()` is exactly the supplied `attrs` reference.

The example explains semantic meaning only. Constructing these values does not inspect an input,
derive a result Shape or layout, construct a Tensor or provenance, read a value, or execute work.

<a id="validation-and-boundaries"></a>
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
described under [scalar select expressions](gather.md#scalar-select-expressions). Gradients, graph/compiler
capture and canonicalization, materialization, backend lowering, and execution remain planned.

<a id="gather-and-gather-elements-semantic-kinds-and-attributes"></a>
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

<a id="one-hot-semantic-kind-and-attributes"></a>
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
[one-hot encoding expressions](gather.md#one-hot-encoding-expressions).

<a id="gather-nd-semantic-kind-and-attributes"></a>
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

<a id="tuple-index-shape-examples"></a>
#### Tuple-index Shape examples

<a id="goal-and-inputs-46"></a>
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

<a id="results-and-interpretation"></a>
##### Results and interpretation

- The first result is `[5, 4]`: indices prefix `[5]` is followed by untouched data suffix `[4]`.
- The second result is `[2, 5, 4]`: indices prefix `[2, 5]` is followed by suffix `[4]`.
- The third result is canonical scalar Shape `[]`, not `[1]`, because both formula terms are
  empty.

The `Operation` in the code example retains exactly `GatherNdKind.GATHER_ND` and the supplied
`attrs` reference. The current zero-batch `Tensor.gatherNd(indices)` convenience uses
`new GatherNdAttrs(0)` rather than a second kind or a default attributes value.

<a id="validation-and-boundaries-1"></a>
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
described under [Gather-ND expressions](scatter.md#gather-nd-expressions). The semantic values themselves do
none of that input-aware work. Gather-ND is distinct from scalar `SELECT`, which stores one
coordinate; axis gather, which indexes one selected axis; and scatter-ND, which writes or combines
updates. Neither semantic values nor current expression construction define index-value bounds,
a numerical algorithm, gradients, compiler behavior, backend behavior, or execution.

<a id="axis-scatter-semantic-kinds-reduction-and-attributes"></a>
### Axis-scatter semantic kinds, reduction, and attributes

`AxisScatterKind` implements `OperationKind` with exactly `SCATTER_ELEMENTS`, then `SCATTER_ADD`.
Both consume ordered logical inputs `[data, indices, updates]` and declare exactly three inputs
and one output. `SCATTER_ELEMENTS` pairs only with `ScatterElementsAttrs(axis, reduction)`;
`SCATTER_ADD` pairs only with `IndexAxisAttrs(axis)`. Cross-family attributes are invalid.

Indices and updates have equal rank and Shape and match data away from the selected axis. The
functional result has the exact data Shape and does not mutate data. For data `[2, 3, 4]`, axis
`1`, and indices/updates `[2, 5, 4]`, update coordinate `[0, 4, 2]` uses its index to select
target `[0, index, 2]`.

`ScatterReduction` contains exactly `NONE`, `ADD`, `MUL`, `MAX`, and `MIN`.
`NONE` means replacement and requires unique targets; the arithmetic values combine the base
and all updates for a target. For `SCATTER_ELEMENTS`, every updates coordinate contributes once
to the target obtained by replacing its selected-axis coordinate with the corresponding index.
The base participates once, duplicates remain distinct contributions, and an unaddressed
coordinate retains the exact base representation. Configurable `MUL`, `MIN`, and `MAX` have the
portable, order-independent represented-value contract specified under
[Scatter Elements expressions](scatter.md#scatter-elements-expressions); the vocabulary selects no
traversal, atomic, tree, or backend algorithm and no derivative or subgradient policy.

`SCATTER_ADD` instead requires updates with the exact Shape that ordinary Gather would produce:
data prefix, complete indices Shape, then data suffix. It has intrinsic addition rather than a
stored reduction. Its data-shaped functional result includes the base and all updates for every
target, so duplicate indices accumulate. `IndexAxisAttrs` is shared with axis Gather because the
same normalized data axis defines both Shape relationships.

`ScatterElementsAttrs` validates the normalized non-negative axis before requiring a non-null
reduction. Public `Tensor.scatterElements` owns caller-axis, type, reduction-eligibility, rank,
Shape, descriptor, and provenance validation. Index bounds and duplicate targets require values
and remain outside model metadata construction. Public `Tensor.scatterAdd` owns the corresponding
numeric and exact Gather-compatible Shape checks. The semantic values define no gradients,
compiler-generated adjoints, materialization, lowering, backend behavior, or execution.

<a id="scatter-nd-semantic-kind-and-attributes"></a>
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

<a id="tuple-index-update-shape-examples"></a>
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
- `MUL`, `MAX`, and `MIN` use the portable, order-independent represented-value contract specified
  under [Scatter Elements expressions](scatter.md#scatter-elements-expressions).

Every tuple position contributes its complete update suffix slice scalar by scalar. The base at
one concrete result coordinate participates exactly once, and every addressed update scalar
participates exactly once. Duplicate tuples remain distinct contributions; an unaddressed
coordinate preserves the exact base representation. These forward semantics select no derivative
or subgradient policy.

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
duplicates, define no derivatives or numerical algorithm, capture no graph, report no backend
support, and execute no work. Scatter-ND differs from Gather-ND, which reads selected data, and
from axis scatter, whose indices address one selected axis rather than multi-axis tuples.


---

[Previous topic](operation-geometry.md) · [Next topic](operation-windows.md)
