# Slices, functional updates and [target](../../glossary.md#term-target)-relative crops

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## The application supplies coordinates; the Tensor method describes a region

Application or layer code calls extraction on `input`, or replacement on base `data` with
an `update` Tensor. Mutable request arrays are cloned, while the exact input/update Tensor
references remain in provenance. No stored values are read, copied or overwritten during
construction. A logical view does not itself attach the input's storage or keep its arena alive.
Compare [scalar select](gather.md#scalar-select-expressions) for removing an axis and
[scatter](scatter.md) when coordinates come from a Tensor rather than Java configuration.

**[Reader model](../../glossary.md#term-reader-model).** A **[slice](../../glossary.md#slice)** selects a [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) range on chosen [axes](../../glossary.md#term-axis).
**[Half-open](../../glossary.md#term-half-open-interval)** means the start is included and the end is excluded; conceptual start `1`,
end `6`, and step `2` select positions `1, 3, 5`. A negative step reverses traversal rather
than making the selected [extent](../../glossary.md#term-extent) negative. A negative axis counts from the final axis; a
negative bound counts relative to the selected axis's extent. These are different roles.
**Functional update** describes a new base-shaped result with a selected region replaced;
it does not mutate the base. A **crop** extracts a region, and its **prefix** states how many
positions precede that region on each axis. [Layout](../../glossary.md#layout) **[stride](../../glossary.md#term-stride)** measures storage-[element](../../glossary.md#term-element) spacing,
whereas slice **step** measures logical-coordinate spacing.

<a id="slice-expressions"></a>
### [Slice](../../glossary.md#slice) expressions

`Tensor.slice(long[] starts, long[] ends, int[] axes, long[] steps)` constructs a fresh general
directional [half-open](../../glossary.md#term-half-open-interval) slice expression. The four arrays are parallel: entry `i` supplies inclusive
raw start `starts[i]`, directional exclusive raw end `ends[i]`, input [axis](../../glossary.md#term-axis) `axes[i]`, and signed
non-zero increment `steps[i]`. Positive steps select `start, start + step, ...` while the [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix)
is below the end; negative steps select while it is above the end. Array lengths must match, and
entry order is preserved. After checking the references and lengths, construction clones every
array before inspecting [elements](../../glossary.md#term-element); it neither mutates nor retains caller-owned arrays.

Each negative axis adds the input [rank](../../glossary.md#term-rank) once and must then be in range. Axes must be distinct after
[normalization](../../glossary.md#term-normalization). Every selected [dimension](../../glossary.md#dimension) must be a `StaticDimension`; an unselected static or
dynamic `Dimension` is retained by exact reference. For [extent](../../glossary.md#term-extent) `D`, a negative bound adds `D`
exactly once. Positive-step start and end then clamp to `[0, D]`. For a negative step and non-zero
`D`, the start clamps to `[0, D - 1]` and the exclusive end to `[-1, D - 1]`. Raw `-1` is still a
relative coordinate and becomes `D - 1`; it is not an end sentinel. Selecting through coordinate
zero therefore needs an end that normalizes below zero, such as raw `-D - 1` when representable.

Checked formulas calculate selected lengths without negating `Long.MIN_VALUE`:

Let integer $s$ be the normalized start, $e$ the directional exclusive end, and $t\ne0$
the signed step. The selected integer count is

$$
\ell=\begin{cases}
1+\lfloor(e-1-s)/t\rfloor&t>0,\ s<e,\\
1+\lfloor(s-1-e)/|t|\rfloor&t<0,\ s>e,\\
0&\text{otherwise}.
\end{cases}
$$

For $s=1$, $e=6$, $t=2$, $\ell=3$ selects $1,3,5$.
For $s=4$, $e=-1$, $t=-1$, $\ell=5$ selects $4,3,2,1,0$ after normalization.
The implementation's negative-step expression is `1 - ((start - 1 - end) / step)`;
it must not negate `Long.MIN_VALUE` to form a primitive absolute value.

Normalized `SliceAttrs` stores `start`, `length`, [normalized axis](../../glossary.md#normalized-axis), and unchanged signed step—not a
normalized end. Empty entries use [canonical](../../glossary.md#term-canonical-form) start zero and length zero. A negative-step selected
zero extent is canonical empty without bound arithmetic. Result rank is unchanged, each selected
axis receives a new static length, and every unselected Dimension reference remains exact.

Four empty arrays are valid and create a fresh explicit [identity](../../glossary.md#term-identity-meanings) slice, including for a [scalar](../../glossary.md#term-scalar).
An empty selection is also valid and produces a zero extent. Empty results deliberately have
unresolved [layout](../../glossary.md#layout): they reference no storage element, so recording one-past-end offset and [stride](../../glossary.md#term-stride)
facts would be arbitrary and could [overflow](../../glossary.md#term-overflow) without adding observable geometry.

When input layout is unresolved, result layout remains unresolved. When input layout is resolved
and the result is non-empty and every step is positive, every resolved input kind—dense
[contiguous](../../glossary.md#term-contiguous), dense with offset, strided, or broadcast zero-stride—produces one new [view](../../glossary.md#term-view-meanings)-marked
logical `LayoutDescriptor`. Starting from the input offset and original strides, entry `i`
performs checked calculations:

Let $o$ be the input element offset, $s_a$ its stride on axis $a$, and let $q$ be the number
of selected entries. Integer entry $0\leq i<q$ names axis $a_i$, normalized start $b_i$
and positive step $t_i$.
For a nonempty resolved result:

$$
o'=o+\sum_{i=0}^{q-1}b_i s_{a_i},\qquad s'_{a_i}=s_{a_i}t_i.
$$

The sum visits each distinct selected axis once; other strides are unchanged. Offsets/strides
are in storage elements, while starts/steps are in logical coordinates.
For row stride $5$, column stride $1$, offset zero and column start one/step two,
the new offset is $1$ and column stride $2$.

`LayoutDescriptor` reclassifies the resulting geometry and calculates its [referenced element
span](../../glossary.md#referenced-element-span). The view flag records logical view [metadata](../../glossary.md#term-metadata) only. Slice construction attaches no [host
storage](../../glossary.md#host-storage) and does not promise a physical [alias](../../glossary.md#aliasing-and-arena), zero-copy route, or executable view.
If any step is negative, the complete result layout is unresolved because the current [descriptor](../../glossary.md#term-descriptor)
forbids negative strides. That boundary chooses neither copying nor a reverse [kernel](../../glossary.md#kernel).

Every successful result preserves the exact input [data type](../../glossary.md#data-type--datatype) and [gradient](../../glossary.md#term-gradient) eligibility, has no
label or host storage, and records `SliceKind.SLICE`, one normalized `SliceAttrs`, one
identity-distinct [producer](../../glossary.md#term-producer) with exact inputs `[input]`, one output descriptor, and [provenance](../../glossary.md#provenance)
output index zero. Identity, repeated, and nested requests remain explicit and consume one fresh
[Tensor](../../glossary.md#tensor) identifier each. [Validation](../../glossary.md#term-validation) and checked arithmetic fail before allocation.

`Tensor.sliceAxis(int, long, long)` uses one entry with step `1L`.
`Tensor.sliceAxis(int, long, long, long)` uses one entry with the caller's signed step.
`Tensor.flip(int... axes)` clones explicit axes in caller order and creates one SLICE producer with
start `D - 1`, length `D`, and step `-1` per non-empty selected extent. A selected zero extent uses
start and length zero. Duplicate normalized axes fail; empty axes mean identity rather than all
axes and remain valid even for a scalar or unselected dynamic dimensions. There is no `FLIP` kind
or per-axis [operation](../../glossary.md#operation) chain.

The official [ONNX Slice contract](https://onnx.ai/onnx/operators/onnx__Slice.html) provides
terminology evidence for signed steps and direction-dependent bound clamping. Synaptik retains
explicit arrays, requires selected static dimensions, and stores selected lengths without [ONNX](../../glossary.md#term-onnx)'s
open-bound sentinel convention. The official [JAX flip reference](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.flip.html)
provides established axis-reversal terminology; Synaptik requires explicit axes and gives empty
varargs identity meaning.

<a id="length-defined-extraction"></a>
#### Length-defined extraction

`Tensor.sliceByLength(long[] starts, long[] lengths, int[] axes, long[] steps)` constructs the
same normalized [`SLICE`](../../glossary.md#slice)/`SliceAttrs` meaning without first interpreting raw [half-open](../../glossary.md#term-half-open-interval) bounds.
The arrays are parallel. Entry `i` directly requests:

$$
c_k=b_i+k\,t_i,\qquad 0\leq k<\ell_i.
$$

Entry $i$ selects an axis, $b_i$ is its supplied start, $t_i$ its non-zero signed step,
$\ell_i\geq0$ its length, and integer $k$ the output coordinate on that axis.
For start $4$, length $3$ and step $-2$, the source coordinates are $4,2,0$.

Starts are already normalized non-negative logical [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix), lengths are finite non-negative
coordinate counts, [axes](../../glossary.md#term-axis) may use caller-facing positive or negative syntax, and steps are signed
and non-zero. No start, length, or bound is clamped, shifted, wrapped, truncated, or inferred.
After null and matching-length checks, all four arrays are cloned in [parameter](../../glossary.md#term-parameter) order and never
retained or mutated.

Entries are validated in caller order. For each entry, axis [normalization](../../glossary.md#term-normalization) and duplicate detection
precede start, length, step, and coordinate checks. For non-zero length, checked arithmetic
calculates `last = start + (length - 1) * step`; both start and last must be non-negative. A static
selected input [extent](../../glossary.md#term-extent) proves both coordinates below its size. If the selected extent is
unresolved, only that upper-bound comparison is deferred. [Model](../../glossary.md#term-model) stores no hidden constraint
object. A zero length still requires a non-negative caller start and non-zero step, but it selects
no coordinate, performs no input-bound proof, and stores [canonical](../../glossary.md#term-canonical-form) start zero.

Result [rank](../../glossary.md#term-rank) is unchanged. Every selected axis receives a new static [Dimension](../../glossary.md#dimension) equal to the
requested length; every unaffected Dimension reference remains exact. The existing [layout](../../glossary.md#layout) rule is
shared with raw slicing: resolved geometry exists only when the result is non-empty, the input
layout is resolved, and every step is positive. Empty results, unresolved input geometry, and any
negative step remain layout-unresolved.

Every success preserves exact input type and [gradient](../../glossary.md#term-gradient) eligibility, has no label or storage, and
records one fresh canonical output at [provenance](../../glossary.md#provenance) index zero with
`SLICE`/`SliceAttrs`/`[input]`. Local [validation](../../glossary.md#term-validation), [Shape](../../glossary.md#shape) calculation, and layout arithmetic precede
the one final [Tensor](../../glossary.md#tensor) identifier allocation.

<a id="complete-slice-expression-example"></a>
#### Complete [slice](../../glossary.md#slice)-expression example

<a id="goal-and-inputs-29"></a>
##### Goal and inputs

Create positive, negative, mixed-direction, empty, zero-[extent](../../glossary.md#term-extent), and flip expressions, then inspect
their normalized [metadata](../../glossary.md#term-metadata). The primary input has resolved [canonical](../../glossary.md#term-canonical-form) [Shape](../../glossary.md#shape) `[2, 5]`; the example
observes [model](../../glossary.md#term-model) metadata only.

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

        // Raw -1 is relative to the extent; -6 is needed to reverse through coordinate zero.
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

<a id="meaningful-lines-and-intermediate-results-14"></a>
##### Meaningful lines and intermediate results

- The positive request selects [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) `[1, 3]`, so it stores start `1`, length `2`, and step
  `2` and derives [Shape](../../glossary.md#shape) `[2, 2]` with resolved [view](../../glossary.md#term-view-meanings) geometry.
- The reverse request selects `[4, 3, 2, 1, 0]`. The explicit raw end `-1` instead normalizes to
  coordinate `4`, so `(4, -1, -1)` is empty and stores [canonical](../../glossary.md#term-canonical-form) start and length zero.
- The mixed request stores both axes in caller order and leaves layout unresolved because one step
  is negative. <span data-glossary-navigation="true"> ([axes glossary](../../glossary.md#term-axis); [layout glossary](../../glossary.md#layout))</span>`flip(1, 0)` stores both negative-step sequences in that same order under one
  producer. The zero-extent flip stores canonical start and length zero.<span data-glossary-navigation="true"> ([producer glossary](../../glossary.md#term-producer); [extent glossary](../../glossary.md#term-extent))</span>
- Every result retains [`FLOAT32`](../../glossary.md#term-float32) and true gradient eligibility and remains unlabeled and
  storage-free.<span data-glossary-navigation="true"> ([gradient glossary](../../glossary.md#term-gradient))</span>

<a id="result-and-interpretation-29"></a>
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

The output demonstrates directional [normalization](../../glossary.md#term-normalization), [canonical](../../glossary.md#term-canonical-form) empty state, exact [Shape](../../glossary.md#shape) [metadata](../../glossary.md#term-metadata),
the positive-[view](../../glossary.md#term-view-meanings)/negative-unresolved [layout](../../glossary.md#layout) boundary, and one-[producer](../../glossary.md#term-producer) flip [provenance](../../glossary.md#provenance). It does
not demonstrate values, physical storage [aliasing](../../glossary.md#term-aliasing), [gradients](../../glossary.md#term-gradient), [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings) or
[canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), [materialization](../../glossary.md#materialization), [backend](../../glossary.md#backend) [lowering](../../glossary.md#lowering), [ONNX](../../glossary.md#term-onnx) mapping, or execution.

<a id="failures-and-useful-variations-26"></a>
##### Failures and useful variations

- A null request array fails with `NullPointerException` naming that parameter. Unequal lengths
  fail before the arrays are cloned or the input descriptor is read.<span data-glossary-navigation="true"> ([parameter glossary](../../glossary.md#term-parameter); [descriptor glossary](../../glossary.md#term-descriptor))</span>
- An axis outside rank after one normalization, the first repeated normalized axis, a zero step,
  or selection of a dynamic dimension fails with <span data-glossary-navigation="true"> ([axis glossary](../../glossary.md#term-axis); [rank glossary](../../glossary.md#term-rank); [normalization glossary](../../glossary.md#term-normalization); [normalized axis glossary](../../glossary.md#normalized-axis); [dimension glossary](../../glossary.md#dimension))</span>`IllegalArgumentException`.
- Every raw `long` bound is accepted, normalized once when negative, and clamped by direction.
  Starts empty in the selected direction produce canonical empty state, not a failure.<span data-glossary-navigation="true"> ([canonical glossary](../../glossary.md#term-canonical-form))</span>
- `flip` rejects its first invalid or duplicate normalized axis and any selected dynamic
  Dimension. Empty flip axes select nothing and are valid identity semantics.<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings))</span>
- Result element-count, offset, stride, layout-classification, or referenced-span overflow
  propagates as <span data-glossary-navigation="true"> ([element glossary](../../glossary.md#term-element); [stride glossary](../../glossary.md#term-stride); [layout glossary](../../glossary.md#layout); [overflow glossary](../../glossary.md#term-overflow))</span>`ArithmeticException`. These local failures occur before Tensor identity
  allocation.<span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor))</span>
- Identifier exhaustion occurs only at final derived construction after normalized attributes,
  Shape, and optional layout have been created.<span data-glossary-navigation="true"> ([Shape glossary](../../glossary.md#shape))</span>

<a id="slice-update-and-target-relative-crop-expressions"></a>
### [Slice update](../../glossary.md#slice-update) and [target-relative crop](../../glossary.md#target-relative-crop) expressions

`Tensor.sliceUpdate(Tensor update, long[] starts, int[] axes, long[] steps)` constructs a fresh
functional replacement. The result has the exact base [Shape](../../glossary.md#shape): [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) selected by the
normalized signed sequences come from `update`, while every coordinate outside that region keeps
the base value. This is neither an in-place write nor an additive scatter.

The three arrays are parallel and are cloned before [descriptor](../../glossary.md#term-descriptor) inspection. Update and base must
have the same [data type](../../glossary.md#data-type--datatype) and [rank](../../glossary.md#term-rank). Each selected update [Dimension](../../glossary.md#dimension) must be static; its size is the
corresponding `SliceAttrs.lengths` entry. Every unselected update Dimension must be structurally
equal to the matching base Dimension. Entry `i` maps update coordinate `k` to:

$$
c_k=b_i+k\,t_i,\qquad 0\leq k<u_{a_i}.
$$

Here entry $i$ selects normalized base axis $a_i$, $b_i$ is normalized start, $t_i\ne0$
the signed step, $u_{a_i}$ the update extent there and $k$ an integer update coordinate.
With $b_i=4$, $t_i=-2$, $u_{a_i}=3$, update positions $0,1,2$ replace base positions
$4,2,0$. No addition or in-place mutation is implied.

[Axes](../../glossary.md#term-axis) normalize once and must be distinct; steps are signed and non-zero. A negative start adds
one static base [extent](../../glossary.md#term-extent) once. For a static base axis, the first and final mapped coordinates must
both be in bounds. An unresolved base axis accepts a non-negative start and defers only the upper-
bound proof. A selected zero update extent uses [canonical](../../glossary.md#term-canonical-form) start zero and maps no coordinates.
Three empty arrays mean explicit full-Shape replacement and still create a fresh [occurrence](../../glossary.md#term-occurrence).

`Tensor.sliceUpdate(Tensor update, Shape prefixShape)` instead places the complete update region
after an exact per-axis prefix. Base, update, and prefix ranks must match, and base/update data
types must be identical. On axis `i`, the replacement interval is:

```text
[prefixShape[i], prefixShape[i] + update.shape[i])
```

The resulting `CropToShapeAttrs` retains the exact update Shape as `targetShape` and the exact
caller prefix Shape. The result retains the exact base Shape; the [target](../../glossary.md#term-target) is the update region, not
the result descriptor. Axes are inspected in order. When base, prefix, and update extents are all
static, checked addition proves `prefix + update <= base`. If any one is unresolved, the whole
axis fit is deferred without partial arithmetic, expression [evaluation](../../glossary.md#term-evaluation), or a [Model](../../glossary.md#term-model)-owned
constraint object. [Scalar](../../glossary.md#term-scalar) replacement and zero update extents are valid when all fully static
fits hold.

`Tensor.cropToShape(Shape targetShape, Shape prefixShape)` constructs target-relative extraction.
The input, target, and prefix ranks must match. On axis `i`, the crop region is:

```text
[prefixShape[i], prefixShape[i] + targetShape[i])
```

The result and `CropToShapeAttrs` retain the exact supplied target Shape reference, while the
attributes also retain the exact prefix Shape reference. A prefix Shape describes non-negative
logical extents preceding the region on each axis; it is not a [Tensor](../../glossary.md#tensor) of indices, a [storage
offset](../../glossary.md#term-storage-offset), or an execution-time bound value. When input, prefix, and target extents are all static,
checked addition proves the region fits. If any extent is unresolved, the inequality remains a
later [binding](../../glossary.md#term-binding) or execution obligation. Construction never clamps, shifts, truncates, or pads the
region.

The operations occupy different parts of the indexing and [layout](../../glossary.md#layout) vocabulary:

| [Operation](../../glossary.md#operation) | Current purpose | Coordinates or Shape source |
|---|---|---|
| [`slice`](../../glossary.md#slice) / `sliceAxis` / `flip` | Extract a raw-bound or reversal region | Caller bounds normalized against selected static input extents |
| `sliceByLength` | Extract a finite normalized sequence, including across an unresolved selected extent | Caller starts/lengths/axes/steps |
| `sliceUpdate(update, starts, axes, steps)` | Functionally replace a signed multi-axis region | Caller starts/axes/steps plus selected static update extents |
| `sliceUpdate(update, prefixShape)` | Functionally replace a complete target-relative update region | Exact update and prefix Shapes, including symbolic Dimensions |
| `cropToShape` | Extract an exact target after a prefix | Exact target and prefix Shapes, including symbolic Dimensions |
| `select` | Extract one scalar coordinate and remove its axis | One scalar index |
| `pad` | Add [constant](../../glossary.md#term-constant)-filled prefix and suffix regions | Rank-sized static widths |
| Scatter families | Replace or combine tensor-indexed targets | Index Tensor values, with family-specific reduction rules |

Every successful update leaves layout unresolved, preserves exact base Shape/type, combines
base/update [gradient](../../glossary.md#term-gradient) eligibility, and records `SliceKind.SLICE_UPDATE`, ordered inputs
`[base, update]`, and output index zero. The array form uses normalized `SliceAttrs`; the
target-relative form uses exact `CropToShapeAttrs`. Every successful crop leaves layout
unresolved, preserves input type/eligibility, and records `SliceKind.SLICE`, exact
`CropToShapeAttrs`, input `[input]`, and output index zero. All results are fresh, canonical,
unlabeled, and storage-free. Construction reads no values, mutates no input, and promises neither
a [view](../../glossary.md#term-view-meanings) nor a copy. Length-defined extraction and target-relative placement are current Model
expressions, and current [compiler](../../glossary.md#term-compiler) [inference](../../glossary.md#term-inference-meanings)/[preflight](../../glossary.md#term-preflight)/[autograd](../../glossary.md#autograd) have adopted their retained
constraints and floating cotangents. [Materialization](../../glossary.md#materialization), [lowering](../../glossary.md#lowering), [backend](../../glossary.md#backend) support, and execution remain outside
the current Model contract.

The official [PyTorch `slice_scatter`](https://docs.pytorch.org/docs/stable/generated/torch.slice_scatter.html)
reference supplies established functional slice-placement terminology. Synaptik's update is one
[metadata](../../glossary.md#term-metadata)-only, signed, multi-axis operation rather than PyTorch's one-dimension call. The official
[JAX `dynamic_update_slice`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html)
also returns an updated array, but its start indices are array/scalar execution values and its
documented fitting behavior may adjust a start; Synaptik stores normalized Java metadata and
rejects a statically out-of-bounds region. The official
[ONNX Slice](https://onnx.ai/onnx/operators/onnx__Slice.html)<span data-glossary-navigation="true"> ([ONNX glossary](../../glossary.md#term-onnx))</span> uses tensor inputs for starts, ends,
axes, and steps. Synaptik's target-relative crop instead stores exact target/prefix Shape metadata
and carries no dynamic bound input Tensor.

<a id="complete-placement-metadata-example"></a>
#### Complete placement-[metadata](../../glossary.md#term-metadata) example

<a id="goal-and-inputs-30"></a>
##### Goal and inputs

Create one signed two-[axis](../../glossary.md#term-axis) update, one fully static crop, and one symbolic crop, then inspect only
their [model](../../glossary.md#term-model) [metadata](../../glossary.md#term-metadata). The update maps [Shape](../../glossary.md#shape) `[3, 2]` into base Shape `[5, 6]`: axis zero visits
[coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) `4, 2, 0`, while axis one visits `1, 3`. The symbolic crop extracts `[N]` after prefix
`[1]` from input Shape `[N + 3]`.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class SlicePlacementExample {
    public static void main(String[] args) {
        Tensor base = tensor(Shape.of(5, 6));
        Tensor update = tensor(Shape.of(3, 2));
        // Functional replacement records a new base-shaped result without writing into base.
        Tensor replaced = base.sliceUpdate(
                update,
                new long[] {-1, 1},
                new int[] {0, 1},
                new long[] {-2, 2});
        SliceAttrs updateAttrs = (SliceAttrs) replaced.provenance().orElseThrow()
                .operation().attrs();

        Shape staticTarget = Shape.of(2, 3);
        Tensor staticCrop = base.cropToShape(staticTarget, Shape.of(1, 2));

        Dimension n = new DynamicDimension("N");
        Tensor symbolicInput = tensor(Shape.ofDimensions(
                DimensionExpressions.addConstant(n, 3)));
        Shape symbolicTarget = Shape.ofDimensions(n);
        Shape symbolicPrefix = Shape.of(1);
        Tensor symbolicCrop = symbolicInput.cropToShape(symbolicTarget, symbolicPrefix);
        CropToShapeAttrs cropAttrs = (CropToShapeAttrs) symbolicCrop.provenance()
                .orElseThrow().operation().attrs();

        System.out.println(replaced.descriptor().shape());
        System.out.println(updateAttrs);
        System.out.println(staticCrop.descriptor().shape() == staticTarget);
        System.out.println(symbolicCrop.descriptor().shape() == symbolicTarget);
        System.out.println(cropAttrs.prefixShape() == symbolicPrefix);
        System.out.println(replaced.provenance().orElseThrow().inputs().get(0) == base);
        System.out.println(replaced.provenance().orElseThrow().inputs().get(1) == update);
        System.out.println(replaced.descriptor().layout().isEmpty());
    }

    private static Tensor tensor(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));
    }
}
```

<a id="meaningful-lines-and-intermediate-results-15"></a>
##### Meaningful lines and intermediate results

- The negative start `-1` on static [extent](../../glossary.md#term-extent) five normalizes to four. With length three and step
  `-2`, the first update [axis](../../glossary.md#term-axis) maps to base [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) `4`, `2`, and `0`.
- Axis one starts at one, has update length two, and uses step two, so it maps to coordinates `1`
  and `3`. `SliceAttrs` retains both normalized sequences in caller order.
- The static crop proves `[1, 1 + 2)` fits axis zero and `[2, 2 + 3)` fits axis one before result
  identity allocation. Its descriptor retains the exact <span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings); [descriptor glossary](../../glossary.md#term-descriptor))</span>`staticTarget` reference.
- The symbolic crop cannot prove `1 + N <= N + 3` locally. It retains the exact target and prefix
  Shape references and defers that inequality without evaluating or simplifying it.<span data-glossary-navigation="true"> ([target glossary](../../glossary.md#term-target); [Shape glossary](../../glossary.md#shape))</span>

<a id="result-and-interpretation-30"></a>
##### Result and interpretation

The program prints:

```text
Shape[5, 6]
SliceAttrs[starts=[4, 1], lengths=[3, 2], axes=[0, 1], steps=[-2, 2]]
true
true
true
true
true
true
```

The output proves [normalization](../../glossary.md#term-normalization), exact [Shape](../../glossary.md#shape)-reference retention, ordered [provenance](../../glossary.md#provenance), and
unresolved result [layout](../../glossary.md#layout). It does not prove replacement or crop values, a physical [alias](../../glossary.md#aliasing-and-arena) or copy,
[compiler](../../glossary.md#term-compiler) adoption, [binding](../../glossary.md#term-binding), [gradients](../../glossary.md#term-gradient), [lowering](../../glossary.md#lowering), or execution.

<a id="failures-and-useful-variations-27"></a>
##### Failures and useful variations

- Slice update rejects mismatched array lengths, type or rank, invalid or duplicate axes, zero
  steps, dynamic selected update extents, negative starts on unresolved base axes, static
  out-of-bounds coordinate sequences, and any complete update Shape mismatch. Checked coordinate
  arithmetic can throw <span data-glossary-navigation="true"> ([Slice update glossary](../../glossary.md#slice-update); [rank glossary](../../glossary.md#term-rank); [axes glossary](../../glossary.md#term-axis); [coordinate glossary](../../glossary.md#axis-rank-scalar-vector-and-matrix); [Shape glossary](../../glossary.md#shape))</span>`ArithmeticException`. These failures consume no [Tensor](../../glossary.md#tensor) identifier.
- Crop rejects target or prefix rank mismatch and the first fully static out-of-bounds region.
  Checked prefix-plus-target overflow throws <span data-glossary-navigation="true"> ([target glossary](../../glossary.md#term-target); [overflow glossary](../../glossary.md#term-overflow))</span>`ArithmeticException`; unresolved inequalities are
  accepted as deferred obligations. These failures also precede result identity allocation.<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings))</span>
- Empty update arrays request full-Shape replacement. A zero selected update extent requests an
  explicit empty placement. Scalar crop uses scalar target and prefix Shapes.<span data-glossary-navigation="true"> ([extent glossary](../../glossary.md#term-extent); [Scalar glossary](../../glossary.md#term-scalar))</span>

<a id="complete-length-defined-and-symbolic-placement-metadata-example"></a>
#### Complete length-defined and symbolic-placement [metadata](../../glossary.md#term-metadata) example

<a id="goal-and-inputs-31"></a>
##### Goal and inputs

Create a static positive-step extraction, a dynamic positive-step extraction, a dynamic
negative-step extraction, a zero-length extraction, and a symbolic [target](../../glossary.md#term-target)-relative placement.
The program inspects [Shapes](../../glossary.md#shape), normalized attributes, [layout](../../glossary.md#layout) state, and exact Shape-reference
retention only.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class DynamicSliceMetadataExample {
    public static void main(String[] args) {
        Shape staticShape = Shape.of(3, 8);
        Tensor staticInput = tensor(
                staticShape, Optional.of(LayoutDescriptor.contiguous(staticShape)));
        // Length is explicit; unresolved input size defers only the upper-bound proof.
        Tensor staticSlice = staticInput.sliceByLength(
                new long[] {1}, new long[] {3}, new int[] {1}, new long[] {2});

        Dimension n = new DynamicDimension("N");
        Tensor dynamicInput = tensor(
                Shape.ofDimensions(n, new StaticDimension(8)), Optional.empty());
        Tensor dynamicSlice = dynamicInput.sliceByLength(
                new long[] {2}, new long[] {3}, new int[] {0}, new long[] {2});
        Tensor negativeSlice = tensor(Shape.ofDimensions(n), Optional.empty()).sliceByLength(
                new long[] {4}, new long[] {3}, new int[] {0}, new long[] {-2});

        Dimension m = new DynamicDimension("M");
        Tensor emptySlice = tensor(
                Shape.ofDimensions(new StaticDimension(0), m), Optional.empty()).sliceByLength(
                new long[] {99}, new long[] {0}, new int[] {1}, new long[] {-1});

        Shape baseShape = Shape.ofDimensions(DimensionExpressions.addConstant(n, 3));
        Shape updateShape = Shape.ofDimensions(n);
        Shape prefixShape = Shape.of(1);
        Tensor placed = tensor(baseShape, Optional.empty())
                .sliceUpdate(tensor(updateShape, Optional.empty()), prefixShape);
        CropToShapeAttrs placedAttrs = (CropToShapeAttrs) placed.provenance().orElseThrow()
                .operation().attrs();

        print(staticSlice);
        print(dynamicSlice);
        print(negativeSlice);
        print(emptySlice);
        System.out.println(placed.descriptor().shape());
        System.out.println(placedAttrs.targetShape() == updateShape);
        System.out.println(placedAttrs.prefixShape() == prefixShape);
        System.out.println(placed.provenance().orElseThrow().inputs().size());
    }

    private static Tensor tensor(Shape shape, Optional<LayoutDescriptor> layout) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, layout, true));
    }

    private static void print(Tensor tensor) {
        System.out.println(tensor.descriptor().shape());
        System.out.println((SliceAttrs) tensor.provenance().orElseThrow().operation().attrs());
        System.out.println(tensor.descriptor().layout().isPresent());
    }
}
```

<a id="meaningful-lines-and-intermediate-results-16"></a>
##### Meaningful lines and intermediate results

| Case | Input [Shape](../../glossary.md#shape) | Public request | Stored attributes | Result Shape | Interpretation |
|---|---|---|---|---|---|
| Static | `[3, 8]` | start `[1]`, length `[3]`, [axis](../../glossary.md#term-axis) `[1]`, step `[2]` | start `[1]`, length `[3]`, axis `[1]`, step `[2]` | `[3, 3]` | [Coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) `[1, 3, 5]` fit statically; resolved positive-step [view](../../glossary.md#term-view-meanings) geometry is available. |
| Dynamic | `[N, 8]` | start `[2]`, length `[3]`, axis `[0]`, step `[2]` | unchanged normalized request | `[3, 8]` | The non-negative coordinates `[2, 4, 6]` are representable; only their upper bound against `N` is deferred. |
| Negative step | `[N]` | start `[4]`, length `[3]`, axis `[0]`, step `[-2]` | unchanged normalized request | `[3]` | Coordinates `[4, 2, 0]` prove the lower bound; the upper bound against `N` is deferred and [layout](../../glossary.md#layout) is unresolved. |
| Zero length | `[0, M]` | start `[99]`, length `[0]`, axis `[1]`, step `[-1]` | start `[0]`, length `[0]`, axis `[1]`, step `[-1]` | `[0, 0]` | No coordinate exists, so start canonicalizes to zero and no bound against `M` is inspected. |
| Symbolic placement | base `[N + 3]`, update `[N]`, prefix `[1]` | `base.sliceUpdate(update, prefix)` | [target](../../glossary.md#term-target) is exact update Shape `[N]`; prefix is exact `[1]` | `[N + 3]` | [Model](../../glossary.md#term-model) retains exact region/base [metadata](../../glossary.md#term-metadata) and defers the unresolved fit; the result is not the target Shape. |

<a id="result-and-interpretation-31"></a>
##### Result and interpretation

The program prints:

```text
Shape[3, 3]
SliceAttrs[starts=[1], lengths=[3], axes=[1], steps=[2]]
true
Shape[3, 8]
SliceAttrs[starts=[2], lengths=[3], axes=[0], steps=[2]]
false
Shape[3]
SliceAttrs[starts=[4], lengths=[3], axes=[0], steps=[-2]]
false
Shape[0, 0]
SliceAttrs[starts=[0], lengths=[0], axes=[1], steps=[-1]]
false
Shape[N + 3]
true
true
2
```

The output proves static requested lengths, deferred dynamic upper bounds, negative-step and
empty-[layout](../../glossary.md#layout) boundaries, zero-length [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), exact symbolic [target](../../glossary.md#term-target)/prefix retention,
base-[Shape](../../glossary.md#shape) result retention, and ordered two-input placement [provenance](../../glossary.md#provenance). It does not prove value
selection or replacement, a physical [view](../../glossary.md#term-view-meanings) or copy, concrete [binding](../../glossary.md#term-binding), numerical [gradients](../../glossary.md#term-gradient),
[lowering](../../glossary.md#lowering), [backend](../../glossary.md#backend) behavior, or execution.

<a id="failures-and-useful-variations-28"></a>
##### Failures and useful variations

- `sliceByLength` null-checks `starts`, `lengths`, [`axes`](../../glossary.md#term-axis), and `steps` in that order after its
  receiver exists. Unequal lengths fail before cloning or descriptor inspection.<span data-glossary-navigation="true"> ([descriptor glossary](../../glossary.md#term-descriptor))</span>
- At each entry, invalid axis and duplicate-axis failures precede negative start, negative length,
  zero step, and coordinate-fit failures. Checked final-coordinate or layout arithmetic may throw<span data-glossary-navigation="true"> ([coordinate glossary](../../glossary.md#axis-rank-scalar-vector-and-matrix); [layout glossary](../../glossary.md#layout))</span>
  `ArithmeticException`. A negative final coordinate fails even for an unresolved selected
  extent. These failures consume no Tensor identifier.<span data-glossary-navigation="true"> ([extent glossary](../../glossary.md#term-extent); [Tensor glossary](../../glossary.md#tensor))</span>
- Target-relative placement checks update type, update rank, prefix rank, then fully static fits.
  Checked <span data-glossary-navigation="true"> ([Target glossary](../../glossary.md#term-target); [rank glossary](../../glossary.md#term-rank))</span>`prefix + update` [overflow](../../glossary.md#term-overflow) throws `ArithmeticException`; any axis with an unresolved
  base, prefix, or update extent defers its whole fit. Scalar Shapes form a valid replacement.<span data-glossary-navigation="true"> ([Scalar glossary](../../glossary.md#term-scalar); [Shapes glossary](../../glossary.md#shape))</span>


---

[Previous topic](axis-transforms.md) · [Next topic](gather.md)
