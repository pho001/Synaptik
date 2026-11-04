# Permute, [transpose](../../glossary.md#term-transpose) and [rank](../../glossary.md#term-rank) editing

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Reordering coordinates is not moving storage

Application or layer-building code calls `input.permute(axes)` with a mutable Java axis array.
The method copies that request, but retains the input Tensor itself as the source description.
For input shape `[2, 3, 4]`, order `[1, 2, 0]` means output coordinate $(j,k,i)$ denotes
input coordinate $(i,j,k)$, where $0\leq i<2$, $0\leq j<3$, and $0\leq k<4$.
The output shape is `[3, 4, 2]`; there are still $2\times3\times4=24$ positions.
This coordinate relationship does not itself copy values or extend the lifetime of borrowed storage.
Use [layout geometry](layouts.md) for offsets/strides and [reshape](reshape-expand.md)
when the goal is regrouping positions rather than changing axis order.

**[Reader model](../../glossary.md#term-reader-model).** An [axis](../../glossary.md#term-axis) is one [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) direction, and rank counts those directions.
**[Permutation](../../glossary.md#permutation)** reorders axes: exchanging rows and columns is a [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) **transpose**.
It does not sort [element](../../glossary.md#term-element) values. Inserting a [singleton](../../glossary.md#term-singleton) axis changes [shape](../../glossary.md#shape) `[2, 3]` to,
for example, `[2, 1, 3]` without adding logical elements; **squeeze** removes such an [extent](../../glossary.md#term-extent)-one
axis. These are different from [expand](reshape-expand.md#expand-expressions), which can repeat
positions. A resolved logical [view](../../glossary.md#term-view-meanings) can describe reordered [strides](../../glossary.md#term-stride) while still leaving the
physical [alias](../../glossary.md#aliasing-and-arena)-or-copy decision to later stages.

<a id="permute-and-transpose-expressions"></a>
### Permute and [transpose](../../glossary.md#term-transpose) expressions

The current `Tensor.permute(int...)` method creates a fresh [model](../../glossary.md#term-model) expression for one complete [axis](../../glossary.md#term-axis)
reordering. The requested array uses output-to-input order: entry `i` identifies the input axis
placed at output axis `i`. Its length must equal input [rank](../../glossary.md#term-rank), and its normalized entries must contain
every input axis exactly once. Each negative entry adds the rank once, so rank-three axes
`[1, -1, 0]` normalize to `[1, 2, 0]`. The method copies the array after count [validation](../../glossary.md#term-validation) and never
retains or mutates caller-owned storage. Empty axes form the valid [rank-zero](../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../glossary.md#term-scalar) [permutation](../../glossary.md#permutation).

The result [Shape](../../glossary.md#shape) contains the exact [immutable](../../glossary.md#term-immutable) input [`Dimension`](../../glossary.md#dimension) references in reordered positions.
If input [layout](../../glossary.md#layout) is resolved, the result receives one new [view](../../glossary.md#term-view-meanings)-marked `LayoutDescriptor` with the
same [element](../../glossary.md#term-element) offset and exact [strides](../../glossary.md#term-stride) in the same reordered positions. Every resolved layout kind
is accepted, and the [descriptor](../../glossary.md#term-descriptor) derives the result kind and [referenced span](../../glossary.md#referenced-element-span) from the new geometry.
If input layout is unresolved, including dynamic Shape cases, result layout remains unresolved.

`Tensor.transpose()` requires rank two and delegates to the same construction with normalized axes
`[1, 0]`. It records `AxisTransformKind.PERMUTE` and `PermutationAttrs([1, 0])`; there is no separate
transpose kind. Both methods preserve exact [data type](../../glossary.md#data-type--datatype) and [gradient](../../glossary.md#term-gradient) eligibility, return a fresh
unlabeled [Tensor](../../glossary.md#tensor) without [host storage](../../glossary.md#host-storage), and record exactly the receiver as their [provenance](../../glossary.md#provenance) input.
[Identity](../../glossary.md#term-identity-meanings), inverse, repeated, and nested requests remain explicit expressions.

Resolved view layout is logical [metadata](../../glossary.md#term-metadata). It does not attach or inspect storage, prove a physical
[alias](../../glossary.md#aliasing-and-arena), promise zero-copy execution, or choose whether a [backend](../../glossary.md#backend) must materialize a copy.
[Compiler](../../glossary.md#term-compiler) [autograd](../../glossary.md#autograd) reverses a supported floating `PERMUTE` with the exact inverse permutation;
qualified backend [owners](../../glossary.md#term-owner) already lower, [prepare](../../glossary.md#prepare) and execute affine view occurrences.
That support does not imply general permutation [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), arbitrary dynamic [admission](../../glossary.md#term-admission),
or a physical alias guarantee from the Model constructor.

<a id="complete-expression-construction-example-4"></a>
#### Complete expression-construction example

<a id="goal-and-inputs-27"></a>
##### Goal and inputs

Permute a resolved [rank](../../glossary.md#term-rank)-three [Tensor](../../glossary.md#tensor) with one negative raw [axis](../../glossary.md#term-axis), then construct a [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix)
[transpose](../../glossary.md#term-transpose) and inspect their normalized semantics. The example observes [descriptors](../../glossary.md#term-descriptor) and [provenance](../../glossary.md#provenance)
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
        // The request is copied; this later mutation must not rewrite the expression.
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

- [Rank](../../glossary.md#term-rank)-three raw [axes](../../glossary.md#term-axis) `[1, -1, 0]` normalize once to `[1, 2, 0]`, so input [Shape](../../glossary.md#shape) `[2, 3, 4]`
  becomes `[3, 4, 2]`. Mutating the caller array later cannot change stored attributes.
- Input [strides](../../glossary.md#term-stride) `[12, 4, 1]` follow the same output-to-input mapping and become `[4, 1, 12]`.
  Offset five remains unchanged. The greatest referenced index is 28, so the referenced element
  span is 29.<span data-glossary-navigation="true"> ([referenced element span glossary](../../glossary.md#referenced-element-span))</span>
- The resolved result is a logical strided view descriptor. The method does not attach the input's
  storage or claim that execution can consume the view without a copy.<span data-glossary-navigation="true"> ([view glossary](../../glossary.md#term-view-meanings); [descriptor glossary](../../glossary.md#term-descriptor))</span>
- [Transpose](../../glossary.md#term-transpose) swaps `[2, 3]` to `[3, 2]`, records normalized `[1, 0]`, and keeps layout unresolved
  because its input layout was unresolved.<span data-glossary-navigation="true"> ([layout glossary](../../glossary.md#layout))</span>

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

The output demonstrates defensive raw-[axis](../../glossary.md#term-axis) [ownership](../../glossary.md#term-ownership-meanings), [normalization](../../glossary.md#term-normalization), [Shape](../../glossary.md#shape)/[stride](../../glossary.md#term-stride) reordering,
offset preservation, exact [provenance](../../glossary.md#provenance), and [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [transpose](../../glossary.md#term-transpose) semantics. It does not demonstrate
value movement, physical storage [aliasing](../../glossary.md#term-aliasing), [gradients](../../glossary.md#term-gradient), [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings) or [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form),
[materialization](../../glossary.md#materialization), [backend](../../glossary.md#backend) [lowering](../../glossary.md#lowering), or execution.

<a id="failures-and-useful-variations-24"></a>
##### Failures and useful variations

- A null [axis](../../glossary.md#term-axis) array fails with `NullPointerException` naming `requestedAxes`.
- An axis count different from rank, a value outside the rank after one negative normalization, or
  the first duplicated normalized axis fails with <span data-glossary-navigation="true"> ([normalization glossary](../../glossary.md#term-normalization); [normalized axis glossary](../../glossary.md#normalized-axis))</span>`IllegalArgumentException`. These failures occur
  before Tensor identity allocation.<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings))</span>
- `transpose()` on any [rank](../../glossary.md#term-rank) other than two fails with `IllegalStateException` reporting the rank.
- Resolved [layout](../../glossary.md#layout) classification or referenced-span [overflow](../../glossary.md#term-overflow) propagates as `ArithmeticException`
  before identity allocation.
- Exhausted [Tensor](../../glossary.md#tensor) identifier space fails with `IllegalStateException` only at final derived
  construction after local immutable metadata has been built.<span data-glossary-navigation="true"> ([immutable glossary](../../glossary.md#term-immutable); [metadata glossary](../../glossary.md#term-metadata))</span>
- `scalar.permute()` is valid. Identity permutations still return fresh expressions rather than
  the receiver.

<a id="expand-dimensions-and-squeeze-expressions"></a>
### Expand-[dimensions](../../glossary.md#dimension) and squeeze expressions

`Tensor.expandDims(int)` inserts one new static dimension of [extent](../../glossary.md#term-extent) one. For input [rank](../../glossary.md#term-rank) `r`, its
caller-facing insertion position is normalized against the result rank `r + 1`: valid raw values
are `[-r - 1, r]`, and a negative value adds `r + 1` once. [Rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) positions `-3` and `0`
therefore insert at the start, while `-1` and `2` insert at the end. Both `-1` and `0` are valid
for a [scalar](../../glossary.md#term-scalar) and produce rank one.

`Tensor.squeeze(int)` instead selects an existing input [axis](../../glossary.md#term-axis) through the ordinary [Shape](../../glossary.md#shape) range
`[-rank, rank - 1]`. The selected dimension must be a `StaticDimension` whose extent is exactly
one. A zero extent, another static extent, or a dynamic dimension is rejected; expression
construction neither guesses a future dynamic [binding](../../glossary.md#term-binding) nor records a [singleton](../../glossary.md#term-singleton) constraint.
Squeezing the only axis of a [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) [Tensor](../../glossary.md#tensor) produces the [canonical](../../glossary.md#term-canonical-form) scalar Shape, while every
axis is invalid for an already scalar Tensor.

Both operations preserve the exact [data type](../../glossary.md#data-type--datatype), [gradient](../../glossary.md#term-gradient) eligibility, and [immutable](../../glossary.md#term-immutable) references and
order of every unaffected Dimension. Every valid call creates a fresh unlabeled, storage-free
Tensor with one-input [provenance](../../glossary.md#provenance). `expandDims` records `AxisTransformKind.EXPAND_DIMS` and an
`AxisTransformAttrs` containing the normalized output insertion position; `squeeze` records
`AxisTransformKind.SQUEEZE` and the normalized input removal position. Repeated, nested, and
inverse-like requests remain explicit and are not canonicalized at this boundary.

Current package-private [compiler](../../glossary.md#term-compiler) [autograd](../../glossary.md#autograd) reverses a supported floating `EXPAND_DIMS` with
`squeeze` at the recorded axis and reverses `SQUEEZE` with `expandDims`. These are logical Tensor
expressions; they do not establish physical [aliasing](../../glossary.md#term-aliasing) or [materialization](../../glossary.md#materialization).

When input [layout](../../glossary.md#layout) is unresolved, result layout stays unresolved. For any resolved input layout,
`expandDims` constructs a new same-offset [view](../../glossary.md#term-view-meanings) [descriptor](../../glossary.md#term-descriptor) by inserting one [stride](../../glossary.md#term-stride) while preserving
all existing strides. The inserted stride is one at the end; otherwise it is the checked product
of the following input stride and extent. `squeeze` constructs a new same-offset view descriptor
by omitting only the selected stride. These descriptors are logical geometry: neither [operation](../../glossary.md#operation)
attaches storage, proves a physical [alias](../../glossary.md#aliasing-and-arena), promises zero-copy execution, or chooses
materialization.

<a id="complete-rank-editing-example"></a>
#### Complete [rank](../../glossary.md#term-rank)-editing example

<a id="goal-and-inputs-28"></a>
##### Goal and inputs

Insert a trailing [singleton](../../glossary.md#term-singleton) into an unresolved-[layout](../../glossary.md#layout) [Tensor](../../glossary.md#tensor) of [Shape](../../glossary.md#shape) `[2, 1, 3]`, and separately
remove its middle singleton. The example observes result Shapes, normalized attributes, and the
storage-free [model](../../glossary.md#term-model) boundary; it does not edit stored values.

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

        // Insertion counts output positions; squeezing selects an existing input axis.
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

- For [rank](../../glossary.md#term-rank) three, insertion [axis](../../glossary.md#term-axis) `-1` adds the result rank `4` and normalizes to output position
  `3`, producing `[2, 1, 3, 1]` and `AxisTransformAttrs[axis=3]`.
- Existing-axis value `-2` adds input rank `3` and normalizes to input position `1`. That
  dimension is statically one, so removal produces <span data-glossary-navigation="true"> ([dimension glossary](../../glossary.md#dimension))</span>`[2, 3]` and
  `AxisTransformAttrs[axis=1]`.
- Both results preserve [`FLOAT32`](../../glossary.md#term-float32) and true gradient eligibility, retain exact one-input
  provenance, and remain unresolved because the input layout is unresolved.<span data-glossary-navigation="true"> ([gradient glossary](../../glossary.md#term-gradient); [provenance glossary](../../glossary.md#provenance); [layout glossary](../../glossary.md#layout))</span>
- Neither result receives a label or host storage. The calls describe rank edits without reading,
  moving, or attaching values.<span data-glossary-navigation="true"> ([host storage glossary](../../glossary.md#host-storage))</span>

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

The output proves the two different [normalization](../../glossary.md#term-normalization) domains, exact [rank](../../glossary.md#term-rank) changes, typed [operation](../../glossary.md#operation)
[identity](../../glossary.md#term-identity-meanings), normalized attributes, unresolved [layout](../../glossary.md#layout) propagation, and absent result storage. It
does not prove value movement, physical [aliasing](../../glossary.md#term-aliasing), [compiler](../../glossary.md#term-compiler) [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), [gradient](../../glossary.md#term-gradient) rules,
[materialization](../../glossary.md#materialization), [backend](../../glossary.md#backend) support, or execution.

<a id="failures-and-useful-variations-25"></a>
##### Failures and useful variations

- An expand-[dimensions](../../glossary.md#dimension) insertion outside `[-rank - 1, rank]` fails with
  `IndexOutOfBoundsException`. An existing squeeze [axis](../../glossary.md#term-axis) outside `[-rank, rank - 1]`, including
  every axis for a scalar, fails through Shape normalization. Both failures occur before result
  identity allocation.<span data-glossary-navigation="true"> ([scalar glossary](../../glossary.md#term-scalar); [Shape glossary](../../glossary.md#shape); [normalization glossary](../../glossary.md#term-normalization))</span>
- Squeezing a zero, non-one, or dynamic dimension fails with `IllegalArgumentException` before
  identity allocation because the singleton fact is not statically proven.<span data-glossary-navigation="true"> ([singleton glossary](../../glossary.md#term-singleton))</span>
- For resolved geometry, insertion-stride multiplication, layout classification, or referenced-
  span overflow fails with <span data-glossary-navigation="true"> ([stride glossary](../../glossary.md#term-stride); [layout glossary](../../glossary.md#layout); [overflow glossary](../../glossary.md#term-overflow))</span>`ArithmeticException` before [identity](../../glossary.md#term-identity-meanings) allocation.
- A resolved canonical, offset, strided, or broadcast input receives new same-offset view
  geometry. Repeated edits and an expand/squeeze pair still return fresh expression identities.<span data-glossary-navigation="true"> ([canonical glossary](../../glossary.md#term-canonical-form); [view glossary](../../glossary.md#term-view-meanings))</span>


---

[Previous topic](reshape-expand.md) · [Next topic](slicing.md)
