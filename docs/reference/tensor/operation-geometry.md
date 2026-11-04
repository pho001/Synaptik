# [Contiguous](../../glossary.md#term-contiguous), reshape, [axis](../../glossary.md#term-axis) and [slice](../../glossary.md#slice) kinds

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**[Reader model](../../glossary.md#term-reader-model).** Geometry describes how logical positions are arranged, not the numerical
values at those positions. **Contiguous** requests dense [row-major](../../glossary.md#term-row-major) placement; **reshape**
regroups the same ordered [elements](../../glossary.md#term-element); **expand** repeats compatible positions; **[permutation](../../glossary.md#permutation)**
reorders axes. A **slice** selects a range, while [slice update](../../glossary.md#slice-update) describes replacement in a new
result. The [operation](../../glossary.md#operation) kind names the request; a [layout](../../glossary.md#layout) classifies resolved [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix)-to-storage
mapping. Thus `CONTIGUOUS` is an operation request, not the same object as a `DENSE_CONTIGUOUS`
layout classification. Read [layouts](layouts.md), [shape transforms](reshape-expand.md)<span data-glossary-navigation="true"> ([shape glossary](../../glossary.md#shape))</span>,
[axis transforms](axis-transforms.md), and [slicing](slicing.md) for the corresponding reader models.

<a id="contiguous-semantic-kind"></a>
### [Contiguous](../../glossary.md#term-contiguous) semantic kind

The public enum
`io.github.pho001.synaptik.model.operation.layout.ContiguousKind` implements [`OperationKind`](../../glossary.md#operationkind) with
exactly one [constant](../../glossary.md#term-constant), `CONTIGUOUS`. It describes one logical input and requests a result with the
same logical values, [Shape](../../glossary.md#shape), [DataType](../../glossary.md#data-type--datatype), and [row-major](../../glossary.md#term-row-major) [element](../../glossary.md#term-element) order in [canonical](../../glossary.md#term-canonical-form) dense row-major
geometry with logical [storage offset](../../glossary.md#term-storage-offset) zero. Input count and these preserved facts are semantic
context; the enum stores no input, [arity](../../glossary.md#term-arity), [descriptor](../../glossary.md#term-descriptor), [layout](../../glossary.md#layout) geometry, or result state.

The kind has no intrinsic [parameters](../../glossary.md#term-parameter) and composes explicitly with the generic descriptor:

```java
Operation contiguous = new Operation(
        ContiguousKind.CONTIGUOUS,
        NoOperationAttrs.INSTANCE);
```

The resulting [`Operation`](../../glossary.md#operation) retains the exact kind and canonical no-attributes [singleton](../../glossary.md#term-singleton). Its
[signature](../../glossary.md#term-signature) enforces that pairing and declares one input and one output.

`ContiguousKind.CONTIGUOUS` is a computation request.
`LayoutKind.DENSE_CONTIGUOUS` instead classifies resolved [strides](../../glossary.md#term-stride) and zero offset for a static
Shape. The semantic kind therefore neither depends on nor replaces the layout classification. It
does not itself inspect whether an input is already contiguous, construct a `LayoutDescriptor`,
return an [alias](../../glossary.md#aliasing-and-arena), allocate or copy storage, derive a [materialization](../../glossary.md#materialization) requirement, select [lowering](../../glossary.md#lowering),
or execute work. Public `Tensor.contiguous()` now supplies the Shape-aware expression boundary:
fully static results receive new canonical geometry, while dynamic results remain unresolved.
[Compiler](../../glossary.md#term-compiler) [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), materialization policy, [planning](../../glossary.md#term-planning), [prepare](../../glossary.md#prepare), [backend](../../glossary.md#backend), [runtime](../../glossary.md#term-runtime), training,
and execution behavior remain with their owning later contracts.

Inherited enum names are diagnostic typed vocabulary, not [serialization](../../glossary.md#term-serialization), [registry](../../glossary.md#term-registry), [dispatch](../../glossary.md#term-dispatch),
[kernel](../../glossary.md#kernel), or reflection identifiers. An equally named constant from another kind family remains a
different typed value.

<a id="reshape-and-expand-semantic-kinds-and-target-shape-attributes"></a>
### Reshape and expand semantic kinds and [target](../../glossary.md#term-target)-[shape](../../glossary.md#shape) attributes

The public enum
`io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind` implements [`OperationKind`](../../glossary.md#operationkind)
with exactly `RESHAPE` and `EXPAND`, in that order. Both meanings have one logical input and one
exact result `Shape`, carried by `TargetShapeAttrs(targetShape)` rather than stored on the enum.

| Kind | Semantic meaning | Deferred input-dependent decision |
|---|---|---|
| `RESHAPE` | Preserve the ordered logical [element](../../glossary.md#term-element) sequence while interpreting it through `targetShape` [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix). | [Graph](../../glossary.md#graph)-wide dynamic count constraints and executable [alias](../../glossary.md#aliasing-and-arena) or copy choice. |
| `EXPAND` | Logically repeat compatible [singleton](../../glossary.md#term-singleton) [dimensions](../../glossary.md#dimension) or add repeated leading dimensions to produce `targetShape`. | [Rank](../../glossary.md#term-rank) and singleton compatibility, symbolic constraints, and zero-[stride](../../glossary.md#term-stride) or materialized geometry. |

The pairings are explicit:

```java
TargetShapeAttrs attrs = new TargetShapeAttrs(Shape.of(3, 2));
Operation reshape = new Operation(ShapeTransformKind.RESHAPE, attrs);
Operation expand = new Operation(ShapeTransformKind.EXPAND, attrs);
```

Both operations retain the exact `attrs` reference. Their shared family [signature](../../glossary.md#term-signature) enforces
`TargetShapeAttrs` and declares one input and one output.
`TargetShapeAttrs` rejects a null target with `NullPointerException("targetShape")`, retains every
non-null [immutable](../../glossary.md#term-immutable) `Shape` reference unchanged, and accepts [scalar](../../glossary.md#term-scalar), zero-[extent](../../glossary.md#term-extent), fully static,
mixed static/dynamic, and fully dynamic Shapes without inspecting an input.

The stored `Shape` is normalized semantic state. Its static dimensions are non-negative and its
dynamic dimensions are explicit symbols, so a raw numeric `-1` reshape-[inference](../../glossary.md#term-inference) sentinel cannot
occur in the attributes. The current raw public reshape boundary normalizes that caller-facing
syntax before constructing this value.

Conceptually, reshaping logical sequence `[a, b, c, d, e, f]` from shape `[2, 3]` to `[3, 2]`
preserves that sequence and changes only its coordinate grouping to `[[a, b], [c, d], [e, f]]`.
Expanding logical row `[a, b, c]` from `[1, 3]` to `[2, 3]` instead repeats it as
`[[a, b, c], [a, b, c]]`. These examples explain the semantic distinction; the implemented enum
and attributes perform neither transformation, compatibility [validation](../../glossary.md#term-validation), [layout](../../glossary.md#layout) construction, nor
value execution. Current `Tensor.reshape` overloads separately perform raw request [normalization](../../glossary.md#term-normalization),
locally provable count validation, result [descriptor](../../glossary.md#term-descriptor)/layout derivation, and [provenance](../../glossary.md#provenance)
construction as described under [Reshape expressions](reshape-expand.md#reshape-expressions). Current
`Tensor.expand` overloads separately validate directional right-aligned compatibility and derive
same-offset zero-stride [view](../../glossary.md#term-view-meanings) geometry when numeric layout facts are resolved, as described under
[Expand expressions](reshape-expand.md#expand-expressions).

These attribute values do not bind dynamic extents, differentiate, plan memory or execute.
[Compiler](../../glossary.md#term-compiler) verification and supported reshape/expand cotangents are implemented; qualified
[backend](../../glossary.md#backend) [owners](../../glossary.md#term-owner) perform preparation and execution under their exact geometry guards.

<a id="axis-transform-semantic-kinds-and-attributes"></a>
### [Axis](../../glossary.md#term-axis)-transform semantic kinds and attributes

The public enum
`io.github.pho001.synaptik.model.operation.layout.AxisTransformKind` implements [`OperationKind`](../../glossary.md#operationkind)
with exactly `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`, in that order. These are one-input semantic
meanings; the enum stores no input, [rank](../../glossary.md#term-rank), [Shape](../../glossary.md#shape), [layout](../../glossary.md#layout), or result state.

| Kind | Semantic meaning | Required attributes |
|---|---|---|
| `PERMUTE` | Reorder every axis through one complete output-to-input mapping. | `PermutationAttrs` |
| `EXPAND_DIMS` | Insert one [extent](../../glossary.md#term-extent)-one axis at a normalized output position. | `AxisTransformAttrs` |
| `SQUEEZE` | Remove one selected extent-one input axis at a normalized input position. | `AxisTransformAttrs` |

<a id="output-to-input-order-example"></a>
#### Output-to-input order example

<a id="goal-and-inputs-41"></a>
##### Goal and inputs

Describe a [permutation](../../glossary.md#permutation) that swaps the conceptual `rows` and `columns` [axes](../../glossary.md#term-axis) while retaining the
`channels` axis, then show all three exact kind/attributes pairings. The input-axis labels are
`[rows, columns, channels]`, and the normalized permutation is `[1, 0, 2]`.

```java
PermutationAttrs permutation = new PermutationAttrs(List.of(1, 0, 2));
AxisTransformAttrs position = new AxisTransformAttrs(1);

Operation permute = new Operation(AxisTransformKind.PERMUTE, permutation);
Operation expandDims = new Operation(AxisTransformKind.EXPAND_DIMS, position);
Operation squeeze = new Operation(AxisTransformKind.SQUEEZE, position);
```

<a id="result-and-interpretation-41"></a>
##### Result and interpretation

`PermutationAttrs.axes()` uses output-to-input order: [element](../../glossary.md#term-element) `i` names the normalized input [axis](../../glossary.md#term-axis)
occupying output position `i`. The mapping therefore yields output-axis labels
`[columns, rows, channels]`. The three [`Operation`](../../glossary.md#operation) values retain their exact supplied attributes.
This example demonstrates semantic [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) order and composition only; it does not construct a
[Tensor](../../glossary.md#tensor), calculate a result [Shape](../../glossary.md#shape) or [layout](../../glossary.md#layout), move values, or execute an operation.

<a id="failure-variation"></a>
##### Failure variation

The [permutation](../../glossary.md#permutation) must contain every integer in `[0, axes.size())` exactly once. An empty list is the
valid [identity](../../glossary.md#term-identity-meanings) permutation for a [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../glossary.md#term-scalar). Construction validates the caller list in
ascending index order and stores an [immutable](../../glossary.md#term-immutable) [snapshot](../../glossary.md#term-snapshot), so later caller mutation cannot change
the attributes and the returned list cannot be mutated. The record proves completeness against
its own list size, not against a future input [Tensor](../../glossary.md#tensor) [rank](../../glossary.md#term-rank).

`AxisTransformAttrs` accepts every non-negative `int`. With `EXPAND_DIMS`, its [axis](../../glossary.md#term-axis) is the output
position where the new [singleton](../../glossary.md#term-singleton) is inserted. With `SQUEEZE`, it is the input position selected
for removal. The record has no input rank or [dimension](../../glossary.md#dimension), so it cannot prove the insertion/removal
bound or that a squeezed dimension has [extent](../../glossary.md#term-extent) one. The current [Shape](../../glossary.md#shape)-aware public expression
boundary owns raw negative-axis [normalization](../../glossary.md#term-normalization) and those input-dependent checks.

A parameterless [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) `transpose()` method is now implemented as a convenience over
`PERMUTE + PermutationAttrs(List.of(1, 0))`; [`TRANSPOSE`](../../glossary.md#term-transpose) is not a fourth semantic kind. Generic
[`Operation`](../../glossary.md#operation) retains the supplied kind and attributes, while family [signatures](../../glossary.md#term-signature) enforce the exact
permutation-versus-axis attributes variants. Public `Tensor.permute` and `transpose` now construct result [descriptors](../../glossary.md#term-descriptor), logical [view](../../glossary.md#term-view-meanings)
[layout](../../glossary.md#layout) when input geometry is resolved, and exact one-input [provenance](../../glossary.md#provenance). Public `expandDims` and
`squeeze` now construct the corresponding rank-edited result descriptors, conditional logical
view layouts, and exact one-input provenance. [Compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings), verification and supported
floating axis-transform cotangents are implemented. Qualified [backend](../../glossary.md#backend) [owners](../../glossary.md#term-owner) lower, [prepare](../../glossary.md#prepare)
and execute the resulting occurrences; the attribute constructors themselves do none of these.

<a id="slice-semantic-kind-and-normalized-attributes"></a>
### [Slice](../../glossary.md#slice) semantic kind and normalized attributes

The public enum `io.github.pho001.synaptik.model.operation.layout.SliceKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly `SLICE`, then `SLICE_UPDATE`. `SLICE` identifies one-input extraction;
`SLICE_UPDATE` identifies two-input functional replacement with ordered inputs `[base, update]`.
`SliceAttrs(starts, lengths, axes, steps)` carries four equal-size parallel lists shared by both
meanings. At entry `i`, it identifies exactly `lengths[i]` [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) beginning at inclusive
`starts[i]` and advancing by signed non-zero `steps[i]` on normalized input [axis](../../glossary.md#term-axis) `axes[i]`.
Coordinate `k` is:

```text
starts[i] + k * steps[i], for 0 <= k < lengths[i]
```

For extraction, an axis without an entry retains its complete logical coordinate range. For
functional replacement, the selected sequences identify where the same-[rank](../../glossary.md#term-rank) update replaces base
coordinates; every other base coordinate remains unchanged. The result has the base [Shape](../../glossary.md#shape), and
the kind defines neither mutation nor addition.

Both kinds accept `CropToShapeAttrs(targetShape, prefixShape)`. Under `SLICE`, the exact [target](../../glossary.md#term-target)
Shape is the extraction result Shape. Under `SLICE_UPDATE`, the exact target Shape is the update
region Shape and the result retains the base Shape. In both forms, the prefix Shape describes the
non-negative logical [extent](../../glossary.md#term-extent) preceding the region on every axis; both exact Shape references are
retained and may contain symbolic [Dimensions](../../glossary.md#dimension). The attributes perform no rank, input, type, bounds,
[binding](../../glossary.md#term-binding), [layout](../../glossary.md#layout), [compiler](../../glossary.md#term-compiler), or execution [validation](../../glossary.md#term-validation).

The exact family [signatures](../../glossary.md#term-signature) are:

| Kind | Exact attributes | Inputs | Outputs |
|---|---|---:|---:|
| `SLICE` | `SliceAttrs` | 1 | 1 |
| `SLICE` | `CropToShapeAttrs` | 1 | 1 |
| `SLICE_UPDATE` | `SliceAttrs` | 2 | 1 |
| `SLICE_UPDATE` | `CropToShapeAttrs` | 2 | 1 |

<a id="parallel-half-open-example"></a>
#### Parallel [half-open](../../glossary.md#term-half-open-interval) example

<a id="goal-and-inputs-42"></a>
##### Goal and inputs

Describe a conceptual mixed-direction [slice](../../glossary.md#slice) of [Shape](../../glossary.md#shape) `[3, 5]` that keeps all three rows and
reverses the columns. The already normalized [parameters](../../glossary.md#term-parameter) are starts `[0, 4]`, lengths `[3, 5]`,
[axes](../../glossary.md#term-axis) `[0, 1]`, and steps `[1, -1]`.

```java
SliceAttrs attrs = new SliceAttrs(
        List.of(0L, 4L),
        List.of(3L, 5L),
        List.of(0, 1),
        List.of(1L, -1L));
Operation slice = new Operation(SliceKind.SLICE, attrs);
```

<a id="meaningful-entries-and-result"></a>
##### Meaningful entries and result

- Entry zero starts at zero, has length three, and steps by one, so its [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) are rows
  `0`, `1`, and `2`.
- Entry one starts at four, has length five, and steps by negative one, so its coordinates are
  columns `4`, `3`, `2`, `1`, and `0`.
- The two entries apply in parallel. `slice.kind()` is exactly `SliceKind.SLICE`, and
  `slice.attrs()` is exactly the supplied `attrs` reference.

The conceptual selected coordinate grid therefore contains all three rows with reversed column
order. This result explains the requested logical selection; constructing the attributes and
[`Operation`](../../glossary.md#operation) does not construct a [Tensor](../../glossary.md#tensor), calculate a result [Shape](../../glossary.md#shape) or [layout](../../glossary.md#layout), read values, or
execute work.

<a id="ownership-validation-and-useful-variations"></a>
##### [Ownership](../../glossary.md#term-ownership-meanings), [validation](../../glossary.md#term-validation), and useful variations

Starts, lengths, and steps use `long`, matching the numeric width of [Shape](../../glossary.md#shape) [dimensions](../../glossary.md#dimension) and [layout](../../glossary.md#layout)
geometry; [axes](../../glossary.md#term-axis) use `int`, matching Java [rank](../../glossary.md#term-rank) and axis positions. Construction requires non-null
equal-size lists and [elements](../../glossary.md#term-element), rejects negative starts, lengths, or axes, repeated axes, and zero
steps. Length zero requires [canonical](../../glossary.md#term-canonical-form) start zero. For positive length, checked arithmetic verifies
that the final selected [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) is representable and non-negative. This admits
`Long.MIN_VALUE` as a step for a one-coordinate sequence. Validation completes before
`List.copyOf` stores one [immutable](../../glossary.md#term-immutable) [snapshot](../../glossary.md#term-snapshot) per component. Caller mutation therefore cannot change
the attributes, accessor mutation fails, and entry order plus all values define equality.

Four empty lists form a normalized [identity](../../glossary.md#term-identity-meanings) [slice](../../glossary.md#slice) that constrains no axes. `SliceAttrs` has no
input Shape, so a non-negative axis may still exceed a future rank. It stores neither raw bounds
nor a direction-dependent end sentinel. Raw negative axes or bounds, clamping, selected-static-
dimension checks, checked length calculation, result Shape and layout, and [provenance](../../glossary.md#provenance) are supplied
by the current public expression boundary described under [Slice expressions](slicing.md#slice-expressions).

The current single-axis convenience uses the same semantic kind with one step-one entry:

```java
new SliceAttrs(
        List.of(normalizedStart),
        List.of(selectedLength),
        List.of(normalizedAxis),
        List.of(1L));
```

The explicit-step single-axis method stores its signed step, and `flip` stores one negative-step
entry per requested axis in the same `SliceAttrs`. `sliceByLength` supplies already normalized
starts and [finite](../../glossary.md#term-finite) lengths through that same semantic value. Neither is a second kind such as
`SLICE_AXIS` or `FLIP`. Public `sliceUpdate` and `cropToShape` supply their distinct input-aware
validation and [metadata](../../glossary.md#term-metadata) construction described under [Slice update and target-relative crop
expressions](slicing.md#slice-update-and-target-relative-crop-expressions)<span data-glossary-navigation="true"> ([Slice update glossary](../../glossary.md#slice-update); [target-relative crop glossary](../../glossary.md#target-relative-crop))</span>. These semantic values define no
storage [view](../../glossary.md#view), [materialization](../../glossary.md#materialization), [gradient](../../glossary.md#term-gradient) or backward scatter, [compiler](../../glossary.md#term-compiler) [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), [ONNX](../../glossary.md#term-onnx)
mapping, [backend route](../../glossary.md#backend-route), or execution behavior.


---

[Previous topic](operation-reductions.md) · [Next topic](operation-indexing.md)
