# Grouped Conv1d, [Conv2d](../../glossary.md#convolution--conv2d) and [Conv3d](../../glossary.md#conv3d--ncdhw)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Which component calls convolution?

Application code or an NN layer builder calls `input.conv2d(weight, bias, attrs)` (or its
rank-specific counterpart). `input` is the source Tensor; `weight` describes the learned
coefficients and `bias` the optional output-channel offsets. This Tensor API retains those
exact references in an expression; it does not take over NN parameter state, copy their values,
or execute a kernel. The owning layer and execution lifecycle are separate from this call.
Use [window extraction and overlap addition](windows.md) to understand spatial sampling and
[matrix contraction](matmul-linear.md) to understand the sum of products.

**[Reader model](../../glossary.md#term-reader-model).** [Convolution](../../glossary.md#term-channels-first-convolution-layers) here applies a small table of [weights](../../glossary.md#term-weight), the **[kernel](../../glossary.md#kernel)**, at successive
[spatial](../../glossary.md#term-spatial-axes) positions. At each output position it multiplies sampled input values by weights and sums
the products; this is a **[contraction](../../glossary.md#contraction-dimension)** over kernel positions and input channels.
A **[channel](../../glossary.md#term-channel)** is one [feature](../../glossary.md#term-feature) plane, and a **[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)** holds independent examples.
**Groups** divide channels into independent subsets, preventing connections across groups.
An optional **[bias](../../glossary.md#term-bias)** adds one value per output channel.

Window **[stride](../../glossary.md#term-stride)** is the distance between window starts, unlike a storage-[layout](../../glossary.md#layout) stride.
**[Dilation](../../glossary.md#term-dilation)** is the spacing between [samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters) inside one kernel; **[padding](../../glossary.md#padding)** extends the conceptual
[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) domain around the input. Kernel size counts samples, while [effective kernel size](../../glossary.md#spatial-axes-channels-and-window-sampling) counts
the full span including dilation gaps. The existing Conv1d diagram below explains composition;
its inserted height-one [axis](../../glossary.md#term-axis) does not add a second varying spatial direction.

<a id="grouped-ncw-conv1d-composition"></a>
### Grouped [NCW](../../glossary.md#term-ncw) [Conv1d composition](../../glossary.md#conv1d-composition)

`input.conv1d(weight, attrs)` and `input.conv1d(weight, bias, attrs)` are [rank](../../glossary.md#term-rank)-specific
conveniences for grouped one-dimensional cross-correlation in NCW order: [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters), [channel](../../glossary.md#term-channel), width.
They add no `CONV1D` [operation](../../glossary.md#operation). Each successful call constructs this exact visible [graph](../../glossary.md#graph):

```text
input  [N, C_in, W]                 --EXPAND_DIMS(2)--\
                                                          CONV2D --SQUEEZE(2)--> [N, C_out, W_out]
weight [C_out, C_in/groups, K_w]    --EXPAND_DIMS(2)--/
optional bias [C_out] -----------------------------------/
```

The expansions produce `[N, C_in, 1, W]` and
`[C_out, C_in/groups, 1, K_w]`. The latter is the rank-three [weight](../../glossary.md#term-weight) with a [singleton](../../glossary.md#term-singleton) [kernel](../../glossary.md#kernel)-height
[axis](../../glossary.md#term-axis) inserted. `Conv1dAttrs(stride, padding, dilation, groups)` maps to the fresh [Conv2d](../../glossary.md#convolution--conv2d) value
`Conv2dAttrs(1, stride, 0, padding, 1, dilation, groups)`: height uses unit [stride](../../glossary.md#term-stride) and [dilation](../../glossary.md#term-dilation)
with zero [padding](../../glossary.md#padding), while width uses the requested geometry. [Bias](../../glossary.md#term-bias) is optional, remains rank one,
and is passed directly to Conv2d without expansion.

For input width `W`, positive static kernel width `K_w`, padding `p`, dilation `d`, and stride
`s`, the result width is:

$$
k_{\mathrm{eff}}=d(K_w-1)+1,\qquad
W_{\mathrm{out}}=\left\lfloor\frac{W+2p-k_{\mathrm{eff}}}{s}\right\rfloor+1.
$$

All sizes count integer positions: $W,p\geq0$ and $K_w,d,s>0$.
The numerator must be non-negative. In the example below, $W=7$, $K_w=3$, $p=1$,
$d=1$, $s=2$, so $k_{\mathrm{eff}}=3$ and $W_{\mathrm{out}}=\lfloor6/2\rfloor+1=4$.

Static invalid geometry and statically provable group, channel, and bias mismatches fail before
the first intermediate is constructed. An unresolved input width retains the same [canonical](../../glossary.md#term-canonical-form)
symbolic expression as Conv2d and defers its non-negative-numerator obligation. Input, weight,
and present bias must be floating. Their ordered promotion, accumulation, conceptual-padding,
special-value, [reassociation](../../glossary.md#term-reassociation), and determinism policies are inherited unchanged from Conv2d.

<a id="biased-grouped-conv1d-metadata-example"></a>
#### Biased grouped Conv1d [metadata](../../glossary.md#term-metadata) example

The goal is to inspect the visible composition. Input `[2, 4, 7]`, [weight](../../glossary.md#term-weight) `[6, 2, 3]`, [bias](../../glossary.md#term-bias)
`[6]`, and two groups use two input channels per group. With [stride](../../glossary.md#term-stride) two, [padding](../../glossary.md#padding) one, and [dilation](../../glossary.md#term-dilation)
one, the result width is four.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

Tensor input = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(2, 4, 7), Optional.empty(), true));
Tensor weight = TensorFactory.create(new TensorDescriptor(
        DataType.BFLOAT16, Shape.of(6, 2, 3), Optional.empty(), false));
Tensor bias = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT64, Shape.of(6), Optional.empty(), false));

// Build the visible rank-edit/Conv2d chain; no input numbers are read here.
Tensor output = input.conv1d(weight, bias, new Conv1dAttrs(2, 1, 1, 2));
var squeeze = output.provenance().orElseThrow();
var conv2d = squeeze.inputs().getFirst().provenance().orElseThrow();

System.out.println(output.descriptor().shape());
System.out.println(output.descriptor().dataType());
System.out.println(squeeze.operation().kind());
System.out.println(conv2d.operation().kind());
System.out.println(conv2d.inputs().get(1).descriptor().shape());
```

It prints:

```text
Shape[2, 6, 4]
FLOAT64
SQUEEZE
CONV2D
Shape[6, 2, 1, 3]
```

The immediate [producer](../../glossary.md#term-producer) is `SQUEEZE`; its input is the [Conv2d](../../glossary.md#convolution--conv2d) result, whose first two inputs come
from distinct `EXPAND_DIMS(2)` producers and whose optional third input is the original bias.
Each call therefore creates four fresh [canonical](../../glossary.md#term-canonical-form) one-output wrappers and contains no
`CONV1D` or `ConvNd` [node](../../glossary.md#node). The example proves current metadata and [provenance](../../glossary.md#provenance) only. It does not
evaluate values, promise [fusion](../../glossary.md#term-fusion) or performance, advertise a [backend capability](../../glossary.md#backend-capability), or execute the
[operation](../../glossary.md#operation).

<a id="grouped-nchw-conv2d-expressions"></a>
### Grouped [NCHW](../../glossary.md#term-nchw) [Conv2d](../../glossary.md#convolution--conv2d) expressions

`input.conv2d(weight, attrs)` and `input.conv2d(weight, bias, attrs)` construct one grouped
two-dimensional cross-correlation expression in NCHW [axis](../../glossary.md#term-axis) order: [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters), input [channel](../../glossary.md#term-channel), height,
width. Cross-correlation means that the [weight](../../glossary.md#term-weight) [kernel](../../glossary.md#kernel) is used in its stored height/width order;
it is not reversed as in the mathematical definition of [convolution](../../glossary.md#term-channels-first-convolution-layers).

The complete [Shape](../../glossary.md#shape) contract is:

| Value | Required Shape | Meaning |
|---|---|---|
| receiver input | `[N, C_in, H, W]` | [rank](../../glossary.md#term-rank)-four NCHW input |
| weight | `[C_out, C_in/groups, K_h, K_w]` | one stored kernel per output channel and input-channel position within its group |
| optional [bias](../../glossary.md#term-bias) | `[C_out]` | one value added to each [spatial](../../glossary.md#term-spatial-axes) result for its output channel |
| result | `[N, C_out, H_out, W_out]` | exact input batch and weight output-channel [Dimensions](../../glossary.md#dimension) plus derived spatial Dimensions |

`Conv2dAttrs` carries only intrinsic geometry and grouping. Kernel extents come from the weight
Shape, and bias presence comes from selecting the three-input overload.

| Attribute | Constraint | Meaning |
|---|---|---|
| `strideHeight`, `strideWidth` | positive `long` | step between adjacent output positions |
| `paddingHeight`, `paddingWidth` | non-negative `long` | conceptual positive-zero positions on each side of the corresponding axis |
| `dilationHeight`, `dilationWidth` | positive `long` | spacing between adjacent stored kernel positions |
| `groups` | positive `long` | number of independent [contiguous](../../glossary.md#term-contiguous) input/output channel groups |

`Conv2dAttrs.defaults()` selects unit [stride](../../glossary.md#term-stride), zero symmetric [padding](../../glossary.md#padding), unit [dilation](../../glossary.md#term-dilation), and one
group. The value owns no [Tensor](../../glossary.md#tensor), Shape, [data type](../../glossary.md#data-type--datatype), [layout](../../glossary.md#layout), storage, [gradient](../../glossary.md#term-gradient), [compiler](../../glossary.md#term-compiler), [backend](../../glossary.md#backend),
or execution state.

For either [spatial axis](../../glossary.md#spatial-axes-channels-and-window-sampling), let `D` be its input [extent](../../glossary.md#term-extent), `k` its positive static kernel extent, `p`
the padding on each side, `d` the dilation, and `s` the stride:

$$
k_{\mathrm{eff}}=d(k-1)+1,\qquad n=D+2p-k_{\mathrm{eff}},\qquad
D_{\mathrm{out}}=\left\lfloor n/s\right\rfloor+1.
$$

Here $k_{\mathrm{eff}}$ is the sampled span, $n$ the available start-distance, and
$D_{\mathrm{out}}$ the output position count. Sizes are integers, $D,p\geq0$,
$k,d,s>0$, and valid geometry requires $n\geq0$.

All literal arithmetic is checked in signed `long`. A static negative numerator fails locally;
zero produces one output position. A dynamic input extent retains the [canonical](../../glossary.md#term-canonical-form) symbolic
equivalent and defers the obligation that the bound numerator be non-negative. Kernel height and
width must themselves be statically known and positive.

Static input and output channel counts must each be divisible by `groups`. When the relevant
Dimensions are static, `weightChannelsPerGroup * groups` must equal `C_in`, and bias length must
equal `C_out`. A relation involving an unresolved Dimension is retained as an obligation for
current package-private compiler verification or later concrete [binding](../../glossary.md#term-binding) because the exact
[descriptors](../../glossary.md#term-descriptor) and attributes remain in [provenance](../../glossary.md#provenance).

<a id="grouped-shape-and-provenance-example"></a>
#### Grouped [Shape](../../glossary.md#shape) and [provenance](../../glossary.md#provenance) example

The goal is to inspect current [metadata](../../glossary.md#term-metadata) for a two-group [convolution](../../glossary.md#term-channels-first-convolution-layers). Input `[2, 4, 7, 7]`, [weight](../../glossary.md#term-weight)
`[6, 2, 3, 3]`, and [bias](../../glossary.md#term-bias) `[6]` use two input channels per group. Unit [stride](../../glossary.md#term-stride) and [dilation](../../glossary.md#term-dilation) with one
[padding](../../glossary.md#padding) position per side preserve the `7 x 7` [spatial](../../glossary.md#term-spatial-axes) Shape.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class GroupedConv2dMetadataExample {
    public static void main(String[] args) {
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 4, 7, 7), Optional.empty(), true));
        Tensor weight = TensorFactory.create(new TensorDescriptor(
                DataType.BFLOAT16, Shape.of(6, 2, 3, 3), Optional.empty(), false));
        Tensor bias = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT64, Shape.of(6), Optional.empty(), false));
        Conv2dAttrs attrs = new Conv2dAttrs(1, 1, 1, 1, 1, 1, 2);

        // Attributes describe sampling and groups, not a selected backend algorithm.
        Tensor output = input.conv2d(weight, bias, attrs);
        var provenance = output.provenance().orElseThrow();

        System.out.println(output.descriptor().dataType());
        System.out.println(output.descriptor().shape());
        System.out.println(output.descriptor().requiresGrad());
        System.out.println(output.descriptor().layout().isEmpty()
                && output.hostStorage().isEmpty());
        System.out.println(provenance.operation().kind());
        System.out.println(provenance.inputs().get(0) == input
                && provenance.inputs().get(1) == weight
                && provenance.inputs().get(2) == bias);
    }
}
```

It prints:

```text
FLOAT64
Shape[2, 6, 7, 7]
true
true
CONV2D
true
```

The result type is [FLOAT64](../../glossary.md#term-float64) because promotion processes input and weight first, then bias. The
example proves Shape, metadata, and ordered provenance only; it does not calculate convolution
values, [capture](../../glossary.md#term-capture-meanings) a [graph](../../glossary.md#graph), construct a [gradient](../../glossary.md#term-gradient), or establish [compiler](../../glossary.md#term-compiler)/[backend](../../glossary.md#backend) support.

<a id="dynamic-height-example"></a>
#### Dynamic-height example

For input [Shape](../../glossary.md#shape) `[N, 4, H, 9]`, [weight](../../glossary.md#term-weight) Shape `[6, 4, 3, 3]`, vertical [stride](../../glossary.md#term-stride) two, vertical [padding](../../glossary.md#padding)
one, vertical [dilation](../../glossary.md#term-dilation) two, and unit horizontal geometry, height becomes:

$$
k_{\mathrm{eff},h}=2(3-1)+1=5,\qquad
H_{\mathrm{out}}=\left\lfloor\frac{H+2-5}{2}\right\rfloor+1
=\left\lfloor\frac{H-3}{2}\right\rfloor+1.
$$

Here $H$ is the eventual integer input height and must be at least $3$; the other
symbols are the effective vertical span and output height. For $H=7$, this gives three
vertical positions, while the horizontal result has seven positions.

The result Shape is `[N, 6, floor((H - 3) / 2) + 1, 7]`. This is an exact symbolic Shape
expression, not a value chosen for `H`. A later [binding](../../glossary.md#term-binding) must reject a concrete `H` for which the
numerator is negative.

<a id="numerical-and-metadata-policy"></a>
#### Numerical and [metadata](../../glossary.md#term-metadata) policy

For output [channel](../../glossary.md#term-channel) `o`, its group selects the corresponding [contiguous](../../glossary.md#term-contiguous) `C_in/groups` input
channels. Each result [element](../../glossary.md#term-element) is the optional [bias](../../glossary.md#term-bias) for `o`, applied exactly once, plus products
over those channels and [kernel](../../glossary.md#kernel) height/width positions in increasing logical-index meaning.
[Coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) outside the input are conceptual [positive zero](../../glossary.md#nan-infinity-and-signed-zero) and still participate in ordinary
[IEEE](../../glossary.md#term-ieee)-754 multiplication. In particular, a padded zero multiplied by an infinite [weight](../../glossary.md#term-weight) follows
ordinary floating behavior rather than being skipped.

Input, weight, and present bias must be floating. Their promoted type is the output type. [FLOAT64](../../glossary.md#term-float64)
output accumulates in FLOAT64. All four low types and [FLOAT32](../../glossary.md#term-float32) output accumulate in FLOAT32, with one
final conversion for a low output. [NaN, infinity, and signed zero](../../glossary.md#nan-infinity-and-signed-zero) follow ordinary multiplication and
addition. An empty channel [contraction](../../glossary.md#contraction-dimension) starts from positive zero before optional bias. Empty
[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) and output-channel [axes](../../glossary.md#term-axis) remain valid. Implementations may reassociate terms and use [fused
multiply-add](../../glossary.md#term-fused-multiply-add-fma), so traversal order, bitwise equality, and identical cross-[backend](../../glossary.md#backend) [rounding](../../glossary.md#term-rounding) are not
promised.

For FLOAT32, exact [CONV2D](../../glossary.md#convolution--conv2d) geometry, conceptual [padding](../../glossary.md#padding), bias, and contributors remain
fixed. Every term participates exactly once under the primitive and aggregate floors. [Convolution](../../glossary.md#term-channels-first-convolution-layers)
does not gain [MATMUL](../../glossary.md#matrix-multiplication--matmul)'s final-[publication](../../glossary.md#term-publication) exact-zero choice or a final-output [tolerance](../../glossary.md#term-tolerance).

Every successful call returns one fresh, unlabeled, storage-free [Tensor](../../glossary.md#tensor) with unresolved [layout](../../glossary.md#layout),
the exact derived [Shape](../../glossary.md#shape), and [gradient](../../glossary.md#term-gradient) eligibility equal to the logical OR of the actual inputs.
Its one-output [provenance](../../glossary.md#provenance) has index zero, retains the exact `Conv2dAttrs` reference, and records
ordered inputs `[input, weight]` or `[input, weight, bias]`. Current construction reads no values
and creates no gradient rule in the [Model](../../glossary.md#term-model). Package-private [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings), [descriptor](../../glossary.md#term-descriptor)-visible
proof or retention of deferred constraints, and compiler-owned first-order input, weight, and
optional bias [cotangent](../../glossary.md#cotangent) construction are current. The compiler formula preserves group isolation
through public unfold, [matrix](../../glossary.md#term-matrix-meanings), reduction, and overlap-accumulating fold expressions.
Qualified Conv2d occurrences and their generated [derivative](../../glossary.md#term-derivative) [topology](../../glossary.md#term-topology) already have backend
[lowering](../../glossary.md#lowering), preparation and execution. Model construction itself performs none of those steps
and grants no arbitrary dynamic convolution, rewrite or [backend route](../../glossary.md#backend-route).

<a id="grouped-ncdhw-conv3d-expressions"></a>
### Grouped [NCDHW](../../glossary.md#conv3d--ncdhw) [Conv3d](../../glossary.md#conv3d--ncdhw) expressions

`input.conv3d(weight, attrs)` and `input.conv3d(weight, bias, attrs)` construct one first-class
grouped three-dimensional cross-correlation expression. NCDHW names the [rank](../../glossary.md#term-rank)-five logical [axis](../../glossary.md#term-axis)
order [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) (`N`), input [channel](../../glossary.md#term-channel) (`C_in`), depth (`D`), height (`H`), then width (`W`).
Cross-correlation uses the stored [kernel](../../glossary.md#kernel) in depth/height/width order without reversing it.

| Value | Required [Shape](../../glossary.md#shape) | Meaning |
|---|---|---|
| receiver input | `[N, C_in, D, H, W]` | rank-five NCDHW input |
| [weight](../../glossary.md#term-weight) | `[C_out, C_in/groups, K_d, K_h, K_w]` | one stored three-dimensional kernel per output channel and input-channel position within its group |
| optional [bias](../../glossary.md#term-bias) | `[C_out]` | one value participating once for each [spatial](../../glossary.md#term-spatial-axes) result in that output channel |
| result | `[N, C_out, D_out, H_out, W_out]` | exact input batch and weight output-channel [Dimensions](../../glossary.md#dimension) plus three derived spatial Dimensions |

`Conv3dAttrs` is an [immutable](../../glossary.md#term-immutable) ten-field value. Its fields remain in depth, height, width order
within each geometry family:

| Fields | Constraint | Meaning |
|---|---|---|
| `strideDepth`, `strideHeight`, `strideWidth` | positive `long` values | steps between adjacent output positions |
| `paddingDepth`, `paddingHeight`, `paddingWidth` | non-negative `long` values | conceptual positive-zero positions on each side of the corresponding axis |
| `dilationDepth`, `dilationHeight`, `dilationWidth` | positive `long` values | spacing between adjacent stored kernel positions |
| `groups` | positive `long` | number of independent [contiguous](../../glossary.md#term-contiguous) input/output channel groups |

`Conv3dAttrs.defaults()` selects unit [stride](../../glossary.md#term-stride) and [dilation](../../glossary.md#term-dilation), zero symmetric [padding](../../glossary.md#padding), and one group.
Kernel extents come from the weight Shape, while choosing the three-input overload supplies bias.

For each [spatial axis](../../glossary.md#spatial-axes-channels-and-window-sampling), let `X` be the input [extent](../../glossary.md#term-extent), `K_x` the positive static kernel extent, `p_x`
the symmetric padding on each side, `d_x` the dilation, and `s_x` the stride:

$$
k_{\mathrm{eff},x}=d_x(K_x-1)+1,\qquad n_x=X+2p_x-k_{\mathrm{eff},x},\qquad
X_{\mathrm{out}}=\left\lfloor n_x/s_x\right\rfloor+1.
$$

Subscript $x$ selects depth, height or width, not a tensor element. Each formula counts
integer positions with $X,p_x\geq0$, $K_x,d_x,s_x>0$ and $n_x\geq0$.
For the depth example below, $X=8$, $K_x=3$, $p_x=1$, $d_x=1$, $s_x=2$
give $n_x=7$ and four output positions.

Literal arithmetic is checked in signed `long`. A static negative numerator fails during [Model](../../glossary.md#term-model)
construction, while zero produces one output position. If an input spatial Dimension is
unresolved, the result retains the [canonical](../../glossary.md#term-canonical-form) symbolic equivalent of the formula; its eventual
concrete value must still make the numerator non-negative. Kernel depth, height, and width must
each be statically known and positive.

Static input and output channel counts must each be divisible by `groups`. When both participating
Dimensions are static, `weightChannelsPerGroup * groups` must equal `C_in`, and bias length must
equal `C_out`. A relation involving an unresolved Dimension is deferred rather than guessed. The
exact inputs, [descriptors](../../glossary.md#term-descriptor), and attributes preserve the facts needed for later proof.

<a id="biased-grouped-conv3d-metadata-example"></a>
#### Biased grouped [Conv3d](../../glossary.md#conv3d--ncdhw) [metadata](../../glossary.md#term-metadata) example

The goal is to inspect current [Model](../../glossary.md#term-model) metadata for two groups. Input `[2, 4, 8, 9, 10]`, [weight](../../glossary.md#term-weight)
`[6, 2, 3, 3, 3]`, and [bias](../../glossary.md#term-bias) `[6]` use two input channels per group. [Strides](../../glossary.md#term-stride) `(2, 1, 2)`, symmetric
[padding](../../glossary.md#padding) `(1, 1, 0)`, and [dilation](../../glossary.md#term-dilation) `(1, 2, 1)` produce [spatial](../../glossary.md#term-spatial-axes) extents `(4, 7, 4)`.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class GroupedConv3dMetadataExample {
    public static void main(String[] args) {
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.BFLOAT16, Shape.of(2, 4, 8, 9, 10), Optional.empty(), true));
        Tensor weight = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(6, 2, 3, 3, 3), Optional.empty(), false));
        Tensor bias = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT64, Shape.of(6), Optional.empty(), false));
        Conv3dAttrs attrs = new Conv3dAttrs(2, 1, 2, 1, 1, 0, 1, 2, 1, 2);

        // Inspect one first-class 3D occurrence; result storage is still absent.
        Tensor output = input.conv3d(weight, bias, attrs);
        var provenance = output.provenance().orElseThrow();

        System.out.println(output.descriptor().shape());
        System.out.println(output.descriptor().dataType());
        System.out.println(output.descriptor().requiresGrad());
        System.out.println(provenance.operation().kind());
        System.out.println(provenance.operation().attrs() == attrs);
        System.out.println(provenance.inputs().get(0) == input
                && provenance.inputs().get(1) == weight
                && provenance.inputs().get(2) == bias);
    }
}
```

It prints:

```text
Shape[2, 6, 4, 7, 4]
FLOAT64
true
CONV3D
true
true
```

The [Shape](../../glossary.md#shape) follows the three formulas independently. Promotion processes input and weight first,
then bias, so the result type is [FLOAT64](../../glossary.md#term-float64). The last three lines identify one `CONV3D` [occurrence](../../glossary.md#term-occurrence),
the exact attributes reference, and exact ordered inputs `[input, weight, bias]`. This example
establishes Model metadata and [provenance](../../glossary.md#provenance) only.

<a id="numerical-and-provenance-policy"></a>
#### Numerical and [provenance](../../glossary.md#provenance) policy

For output [channel](../../glossary.md#term-channel) `o`, its group selects the corresponding [contiguous](../../glossary.md#term-contiguous) `C_in/groups` input
channels. Products are conceptually visited by increasing input channel, [kernel](../../glossary.md#kernel) depth, kernel
height, then kernel width. Optional [bias](../../glossary.md#term-bias) participates exactly once in the accumulation domain
before final output conversion. [Coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) outside the input are conceptual positive-zero
[samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters) and participate in ordinary [IEEE](../../glossary.md#term-ieee)-754 multiplication, including multiplication by
[infinity](../../glossary.md#term-infinity).

Input, [weight](../../glossary.md#term-weight), and present bias must be floating. Promotion processes input and weight first,
then present bias. [FLOAT64](../../glossary.md#term-float64) output accumulates in FLOAT64; all four low types and [FLOAT32](../../glossary.md#term-float32) output
accumulate in FLOAT32, with one final conversion for a low output. [NaN, infinity, and signed zero](../../glossary.md#nan-infinity-and-signed-zero) otherwise follow
ordinary multiplication and addition. An empty input-channel [contraction](../../glossary.md#contraction-dimension) starts at [positive
zero](../../glossary.md#nan-infinity-and-signed-zero) before optional bias; empty [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) and output-channel [axes](../../glossary.md#term-axis) are valid. [Reassociation and fused
multiply-add](../../glossary.md#rounding-sites-fma-and-aggregate-floor) are permitted, so no fixed summation order or bitwise-identical [rounding](../../glossary.md#term-rounding) is promised.

For FLOAT32, exact [CONV3D](../../glossary.md#conv3d--ncdhw) geometry, conceptual [padding](../../glossary.md#padding), bias, and contributors remain
fixed. Every term participates exactly once under the primitive and aggregate floors. [Convolution](../../glossary.md#term-channels-first-convolution-layers)
does not gain [MATMUL](../../glossary.md#matrix-multiplication--matmul)'s final-[publication](../../glossary.md#term-publication) exact-zero choice or a final-output [tolerance](../../glossary.md#term-tolerance).

Each successful call creates one fresh [canonical](../../glossary.md#term-canonical-form) output at [producer](../../glossary.md#term-producer) index zero. The result is
unlabeled and storage-free, has unresolved [layout](../../glossary.md#layout), retains the exact derived [Shape](../../glossary.md#shape) and promoted
type, and combines the input [gradient](../../glossary.md#term-gradient)-request [metadata](../../glossary.md#term-metadata) with logical OR. Provenance retains the
exact `Conv3dAttrs` reference and ordered inputs `[input, weight]` or
`[input, weight, bias]`. Current [Model](../../glossary.md#term-model) construction reads no [tensor](../../glossary.md#tensor) values. [Compiler](../../glossary.md#term-compiler) independently
infers and final-validates these [descriptors](../../glossary.md#term-descriptor) and deferred channel/[spatial](../../glossary.md#term-spatial-axes) relations. Its shared
[Conv2d](../../glossary.md#convolution--conv2d)/Conv3d first-order rule retains groups through unfold/MATMUL/fold/reductions, restores
each selected input/weight/bias [target](../../glossary.md#term-target)'s exact Shape/type, and accumulates overlapping input
contributions through the public fold. Actual execution is qualified for static [CPU](../../glossary.md#term-cpu) FLOAT32/FLOAT64
and [Metal](../../glossary.md#term-metal) FLOAT32, not ordinary dynamic/named-B Conv3d, low CPU arithmetic or arbitrary layouts;
see the [production training boundary](../../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary).


---

[Previous topic](attention.md) · [Next topic](pooling.md)
