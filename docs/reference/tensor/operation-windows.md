# Window-transform kinds and geometry

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Attributes describe where samples would come from

Tensor expression helpers call the window-attribute constructors with primitive geometry and,
for fold, an explicit target Shape. The attributes retain immutable window/shape/scalar references;
no input array or output buffer is supplied. `Operation` checks the matching kind and attribute
class. [Tensor window calls](windows.md) then compare actual operand descriptors and derive
the output shape. This boundary cannot choose a backend or prove that an output is a physical view.

**[Reader model](../../glossary.md#term-reader-model).** A sliding window is a neighborhood sampled at successive [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) positions.
**Unfold** exposes [samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters) as windows or columns; **fold** adds their contributions back at
destination coordinates. Overlap addition means fold is not generally a lossless inverse.
[Window stride](../../glossary.md#term-window-stride) measures movement between window starts, while [dilation](../../glossary.md#term-dilation) measures spacing within a
window; neither is an [element stride](../../glossary.md#element-stride) in storage. The attributes hold those geometric choices,
not input values. Use the [window-transform reader model and overlap diagram](windows.md)
to visualize the relationship before reading the exact kind/attribute pairings below.

<a id="window-transform-semantic-kinds-and-attributes"></a>
### Window-transform semantic kinds and attributes

The public enum
`io.github.pho001.synaptik.model.operation.layout.WindowTransformKind` implements [`OperationKind`](../../glossary.md#operationkind)
with exactly `UNFOLD_AXIS`, `FOLD_AXIS`, `UNFOLD2D`, `FOLD2D`, `UNFOLD3D`, and `FOLD3D`, in that
order. These
[backend](../../glossary.md#backend)-independent meanings materialize sliding windows or scatter-add window contributions;
they do not construct [Tensors](../../glossary.md#tensor), calculate result [Shapes](../../glossary.md#shape), or execute values.

| Kind | Semantic meaning | Required attributes |
|---|---|---|
| `UNFOLD_AXIS` | Materialize no-[padding](../../glossary.md#padding), no-[dilation](../../glossary.md#term-dilation) windows along one normalized general [axis](../../glossary.md#term-axis); replace that [extent](../../glossary.md#term-extent) with window positions and append window size as the final axis. | `UnfoldAxisAttrs` |
| `FOLD_AXIS` | Interpret the eventual input's final [dimension](../../glossary.md#dimension) as window size, remove it, and scatter-add windows along one normalized [target](../../glossary.md#term-target) axis with an explicit restored extent. | `FoldAxisAttrs` |
| `UNFOLD2D` | Convert one conceptual [rank](../../glossary.md#term-rank)-four [NCHW](../../glossary.md#term-nchw) image tensor to [canonical](../../glossary.md#term-canonical-form) rank-three [im2col](../../glossary.md#term-im2col) columns, using direct positive-zero or explicit exact typed padding. | `Window2dAttrs` or `Unfold2dAttrs` |
| `FOLD2D` | Accumulate canonical rank-three columns through [col2im](../../glossary.md#term-col2im) into one explicit rank-four NCHW result Shape. | `Fold2dAttrs` |
| `UNFOLD3D` | Convert one conceptual rank-five [NCDHW](../../glossary.md#conv3d--ncdhw) volume to canonical rank-three volumetric columns, using direct positive-zero or explicit exact typed padding. | `Window3dAttrs` or `Unfold3dAttrs` |
| `FOLD3D` | Accumulate canonical rank-three volumetric columns into one explicit rank-five NCDHW result Shape. | `Fold3dAttrs` |

NCHW orders axes as [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters), [channel](../../glossary.md#term-channel), height, and width. Im2col places sampled image windows into
columns; col2im scatters those column entries back to image [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) and adds entries that
target the same coordinate. These names explain the `UNFOLD2D` and `FOLD2D` meanings; they are
not additional [operation](../../glossary.md#operation) kinds.

<a id="general-axis-unfold-and-fold"></a>
#### General-[axis](../../glossary.md#term-axis) [unfold and fold](../../glossary.md#window-transform)

<a id="goal-and-inputs-47"></a>
##### Goal and inputs

Describe an [axis](../../glossary.md#term-axis) unfold for conceptual input [Shape](../../glossary.md#shape) `[2, 5, 3]`, then the matching one-dimensional
scatter-add behavior independently. The unfold uses [normalized axis](../../glossary.md#normalized-axis) `1`, window size `3`, and
step `1`. The fold uses conceptual window input Shape `[3, 3]`, normalized [target](../../glossary.md#term-target) axis `0`,
explicit output size `5`, and step `1`.

```java
// Output size is explicit because window count alone cannot describe uncovered trailing positions.
UnfoldAxisAttrs unfoldAttrs = new UnfoldAxisAttrs(1, 3, 1);
FoldAxisAttrs foldAttrs = new FoldAxisAttrs(0, 5, 1);

Operation axisUnfold = new Operation(
        WindowTransformKind.UNFOLD_AXIS,
        unfoldAttrs);
Operation axisFold = new Operation(
        WindowTransformKind.FOLD_AXIS,
        foldAttrs);
```

<a id="result-and-interpretation-45"></a>
##### Result and interpretation

For the unfold, the selected [extent](../../glossary.md#term-extent) `5` has three window starts. Its conceptual result [Shape](../../glossary.md#shape) is
`[2, 3, 3, 3]`: [axis](../../glossary.md#term-axis) `1` becomes the three-position window count, the former trailing extent `3`
stays in place, and the window size `3` is appended as the new final axis. For a static selected
extent `D`, current public Shape construction calculates:

$$
Q=\left\lfloor\frac{D-k}{s}\right\rfloor+1.
$$

Here integer $D$ is input extent, $k>0$ window size, $s>0$ step, and $Q$ the window
count, with $D\geq k$. In this example $D=5$, $k=3$, $s=1$, so $Q=3$ starts.

It first must prove `size <= D` and use checked `long` arithmetic. `UnfoldAxisAttrs` implements
neither check nor calculation. General-axis unfold has no image, [padding](../../glossary.md#padding), or [dilation](../../glossary.md#term-dilation) assumption,
and its meaning is materialized windows rather than a promise that a storage [view](../../glossary.md#view) exists.

For the fold, suppose the conceptual `[3, 3]` windows contain
`[[1, 2, 3], [4, 5, 6], [7, 8, 9]]`. Window [element](../../glossary.md#term-element) `offset` from window `windowIndex` [targets](../../glossary.md#term-target)
`windowIndex * step + offset`. Scatter-add therefore gives conceptual Shape `[5]` with values
`[1, 6, 15, 14, 9]`. Overlaps sum, and valid target positions receiving no contribution remain
zero. `outputSize` is explicit because window count, final-[dimension](../../glossary.md#dimension) window size, and step do not
identify trailing positions that the windows did not cover.

The input's final dimension supplies `FOLD_AXIS` window size; `FoldAxisAttrs` deliberately does
not duplicate it. Public `Tensor.foldAxis` normalizes the caller axis against target [rank](../../glossary.md#term-rank), checks
numeric type and static geometry, and creates one fresh expression. This public transformation
does not implement [autograd](../../glossary.md#autograd) or [compiler](../../glossary.md#term-compiler)-generated use.

<a id="validation-and-ownership"></a>
##### [Validation](../../glossary.md#term-validation) and [ownership](../../glossary.md#term-ownership-meanings)

Both records store an already normalized non-negative [`axis`](../../glossary.md#term-axis). Public `Tensor.unfold` accepts a
negative axis while it has the source [rank](../../glossary.md#term-rank) available to normalize and bounds-check the request;
public `Tensor.foldAxis` does the same against the [target](../../glossary.md#term-target) rank, which is input rank minus one.
`UnfoldAxisAttrs` requires positive `size` and `step`. `FoldAxisAttrs` accepts non-negative
`outputSize`, including zero, and requires positive `step`. Both check components in declaration
order and retain every valid primitive unchanged, including their maximum values. They contain no
rank or [Shape](../../glossary.md#shape), so construction cannot prove axis bounds, size fit, input-window compatibility, or
arithmetic representability.

<a id="nchw-im2col-and-overlap-summing-col2im"></a>
#### [NCHW](../../glossary.md#term-nchw) [im2col](../../glossary.md#term-im2col) and overlap-summing [col2im](../../glossary.md#term-col2im)

<a id="goal-and-inputs-48"></a>
##### Goal and inputs

Describe both `UNFOLD2D` attribute variants for conceptual [NCHW](../../glossary.md#term-nchw) [Shape](../../glossary.md#shape) `[1, 1, 3, 3]` with a
2-by-2 [kernel](../../glossary.md#kernel), unit [stride](../../glossary.md#term-stride), zero symmetric [padding](../../glossary.md#padding), unit [dilation](../../glossary.md#term-dilation), and floor mode, then pair the
same window geometry with an explicit fold output Shape.

```java
// Keep padding value separate from window geometry; both are immutable descriptions.
Window2dAttrs window = new Window2dAttrs(
        2, 2,
        1, 1,
        0, 0,
        1, 1,
        false);
Unfold2dAttrs negativeInfinity = new Unfold2dAttrs(
        window, ScalarValue.float32(Float.NEGATIVE_INFINITY));
Fold2dAttrs fold = new Fold2dAttrs(Shape.of(1, 1, 3, 3), window);

Operation imageUnfold = new Operation(
        WindowTransformKind.UNFOLD2D,
        window);
Operation paddedImageUnfold = new Operation(
        WindowTransformKind.UNFOLD2D,
        negativeInfinity);
Operation imageFold = new Operation(
        WindowTransformKind.FOLD2D,
        fold);
```

<a id="geometry-and-result"></a>
##### Geometry and result

[Stride](../../glossary.md#term-stride) is the positive distance between consecutive window starts. [Dilation](../../glossary.md#term-dilation) is the positive
spacing between adjacent [kernel](../../glossary.md#kernel) [samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters). Symmetric [padding](../../glossary.md#padding) adds the same non-negative logical
width on both sides of a [spatial](../../glossary.md#term-spatial-axes) [dimension](../../glossary.md#dimension). Direct `UNFOLD2D + Window2dAttrs` treats sampled
positions outside the source as conceptual positive zeros. `UNFOLD2D + Unfold2dAttrs` instead uses
its exact typed [scalar](../../glossary.md#term-scalar) for symmetric padding and terminal ceil-grid samples beyond the padded
[extent](../../glossary.md#term-extent). Effective kernel is the spatial span covered after dilation. For
height and width independently, current public static-or-symbolic [Shape](../../glossary.md#shape) construction uses:

For either spatial axis, let $D\geq0$ be input extent, $p\geq0$ padding per side,
$k>0$ kernel count, $d>0$ dilation and $s>0$ window stride, all integers:

$$
k_{\mathrm{eff}}=d(k-1)+1,\quad n=D+2p-k_{\mathrm{eff}},\quad
Q_{\mathrm{floor}}=\lfloor n/s\rfloor+1,\quad
Q_{\mathrm{ceil}}=\lceil n/s\rceil+1.
$$

Valid geometry requires $n\geq0$; $Q$ counts starts, not samples per window.
For $D=3$, $k=2$, $p=0$, $d=s=1$, both modes give two starts on each axis.

Floor mode rounds the quotient down; ceil mode rounds it up. Current public expression
construction folds static calculations with checked `long` arithmetic and requires the effective
kernel to fit when statically provable. Dynamic spatial Dimensions retain [canonical](../../glossary.md#term-canonical-form) [linear](../../glossary.md#term-linear-meanings) and
division expressions; multiplying height and width position counts uses a canonical symbolic
product. `Window2dAttrs` stores the exact geometry and [rounding](../../glossary.md#term-rounding) flag but evaluates no formula.

The example has output height and width `2`. Canonical [im2col](../../glossary.md#term-im2col) Shape is
`[N, C * kernelHeight * kernelWidth, outputHeight * outputWidth]`, so the conceptual result is
`[1, 4, 4]`. Folding compatible columns into explicit Shape `[1, 1, 3, 3]` scatter-adds repeated
[coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix): the center receives four contributions and each corner receives one. [Col2im](../../glossary.md#term-col2im) does not
divide by overlap count, and uncovered output positions remain zero.

<a id="structural-boundary"></a>
##### Structural boundary

`Window2dAttrs` requires positive [kernel](../../glossary.md#kernel), [stride](../../glossary.md#term-stride), and [dilation](../../glossary.md#term-dilation) [dimensions](../../glossary.md#dimension) and non-negative
[padding](../../glossary.md#padding) dimensions, validating them in component order. It retains valid `long` values and
`ceilMode` unchanged without arithmetic. `Unfold2dAttrs` null-checks `window` before
`paddingValue`, retains both exact references and [scalar](../../glossary.md#term-scalar) bits, and performs no input-aware type or
geometry [validation](../../glossary.md#term-validation). `Fold2dAttrs` null-checks `outputShape` before `window` and retains both exact
[immutable](../../glossary.md#term-immutable) references. The structural record accepts every current [Shape](../../glossary.md#shape)
category, including scalar, non-[rank](../../glossary.md#term-rank)-four, zero-[extent](../../glossary.md#term-extent), and dynamic Shapes, because it has no input
columns to compare. Current public `fold2d` construction establishes rank-four [NCHW](../../glossary.md#term-nchw) and exact
static-or-symbolic structural compatibility before constructing a fold expression. Unrelated
symbols are rejected rather than registered as equality constraints.

The 3D records follow the same structural boundary. `Window3dAttrs` contains depth-height-width
kernel, stride, symmetric padding, dilation, and literal-ceil selection in that order.
`Unfold3dAttrs` retains the exact window and typed padding references. `Fold3dAttrs` retains the
exact [NCDHW](../../glossary.md#conv3d--ncdhw) [target](../../glossary.md#term-target) Shape and window references. They perform no input-aware rank, type, fit,
product, column-compatibility, or arithmetic validation.

Family [signatures](../../glossary.md#term-signature) enforce the eight exact attributes pairings and declare one input and one
output; each unfold kind lists direct window attributes before explicit-padding attributes.
The semantic values themselves define no [Tensor](../../glossary.md#tensor) construction,
result-data-type rules, resolved [layout](../../glossary.md#layout), storage, [provenance](../../glossary.md#provenance), [gradients](../../glossary.md#term-gradient), [graph](../../glossary.md#graph)/[compiler](../../glossary.md#term-compiler) behavior,
[planning](../../glossary.md#term-planning), [prepare](../../glossary.md#prepare), [runtime](../../glossary.md#term-runtime), [backend](../../glossary.md#backend) or [ONNX](../../glossary.md#term-onnx) behavior, [materialization](../../glossary.md#materialization), or execution. The current
public Tensor methods add only the local [descriptor](../../glossary.md#term-descriptor) and provenance behavior documented below.


---

[Previous topic](operation-indexing.md) · [Next topic](operation-elementwise.md)
