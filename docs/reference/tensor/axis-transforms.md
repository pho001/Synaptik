# Permute, transpose and rank editing

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="permute-and-transpose-expressions"></a>
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
permutation canonicalization, planning and prepare-time materialization, backend lowering,
runtime residency, and execution remain planned in their owning layers. Current package-private
compiler autograd reverses a supported floating `PERMUTE` with the exact inverse permutation.

<a id="complete-expression-construction-example-4"></a>
#### Complete expression-construction example

<a id="goal-and-inputs-27"></a>
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

<a id="meaningful-lines-and-intermediate-results-12"></a>
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

<a id="result-and-interpretation-27"></a>
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

<a id="failures-and-useful-variations-24"></a>
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

<a id="expand-dimensions-and-squeeze-expressions"></a>
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

Current package-private compiler autograd reverses a supported floating `EXPAND_DIMS` with
`squeeze` at the recorded axis and reverses `SQUEEZE` with `expandDims`. These are logical Tensor
expressions; they do not establish physical aliasing or materialization.

When input layout is unresolved, result layout stays unresolved. For any resolved input layout,
`expandDims` constructs a new same-offset view descriptor by inserting one stride while preserving
all existing strides. The inserted stride is one at the end; otherwise it is the checked product
of the following input stride and extent. `squeeze` constructs a new same-offset view descriptor
by omitting only the selected stride. These descriptors are logical geometry: neither operation
attaches storage, proves a physical alias, promises zero-copy execution, or chooses
materialization.

<a id="complete-rank-editing-example"></a>
#### Complete rank-editing example

<a id="goal-and-inputs-28"></a>
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

<a id="meaningful-lines-and-intermediate-results-13"></a>
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

<a id="result-and-interpretation-28"></a>
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

<a id="failures-and-useful-variations-25"></a>
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


---

[Previous topic](reshape-expand.md) · [Next topic](slicing.md)
