# Grouped Conv1d, Conv2d and Conv3d

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="grouped-ncw-conv1d-composition"></a>
### Grouped NCW Conv1d composition

`input.conv1d(weight, attrs)` and `input.conv1d(weight, bias, attrs)` are rank-specific
conveniences for grouped one-dimensional cross-correlation in NCW order: batch, channel, width.
They add no `CONV1D` operation. Each successful call constructs this exact visible graph:

```text
input  [N, C_in, W]                 --EXPAND_DIMS(2)--\
                                                          CONV2D --SQUEEZE(2)--> [N, C_out, W_out]
weight [C_out, C_in/groups, K_w]    --EXPAND_DIMS(2)--/
optional bias [C_out] -----------------------------------/
```

The expansions produce `[N, C_in, 1, W]` and
`[C_out, C_in/groups, 1, K_w]`. The latter is the rank-three weight with a singleton kernel-height
axis inserted. `Conv1dAttrs(stride, padding, dilation, groups)` maps to the fresh Conv2d value
`Conv2dAttrs(1, stride, 0, padding, 1, dilation, groups)`: height uses unit stride and dilation
with zero padding, while width uses the requested geometry. Bias is optional, remains rank one,
and is passed directly to Conv2d without expansion.

For input width `W`, positive static kernel width `K_w`, padding `p`, dilation `d`, and stride
`s`, the result width is:

```text
effectiveKernel = d * (K_w - 1) + 1
W_out           = floor((W + 2 * p - effectiveKernel) / s) + 1
```

Static invalid geometry and statically provable group, channel, and bias mismatches fail before
the first intermediate is constructed. An unresolved input width retains the same canonical
symbolic expression as Conv2d and defers its non-negative-numerator obligation. Input, weight,
and present bias must be floating. Their ordered promotion, accumulation, conceptual-padding,
special-value, reassociation, and determinism policies are inherited unchanged from Conv2d.

<a id="biased-grouped-conv1d-metadata-example"></a>
#### Biased grouped Conv1d metadata example

The goal is to inspect the visible composition. Input `[2, 4, 7]`, weight `[6, 2, 3]`, bias
`[6]`, and two groups use two input channels per group. With stride two, padding one, and dilation
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

The immediate producer is `SQUEEZE`; its input is the Conv2d result, whose first two inputs come
from distinct `EXPAND_DIMS(2)` producers and whose optional third input is the original bias.
Each call therefore creates four fresh canonical one-output wrappers and contains no
`CONV1D` or `ConvNd` node. The example proves current metadata and provenance only. It does not
evaluate values, promise fusion or performance, advertise a backend capability, or execute the
operation.

<a id="grouped-nchw-conv2d-expressions"></a>
### Grouped NCHW Conv2d expressions

`input.conv2d(weight, attrs)` and `input.conv2d(weight, bias, attrs)` construct one grouped
two-dimensional cross-correlation expression in NCHW axis order: batch, input channel, height,
width. Cross-correlation means that the weight kernel is used in its stored height/width order;
it is not reversed as in the mathematical definition of convolution.

The complete Shape contract is:

| Value | Required Shape | Meaning |
|---|---|---|
| receiver input | `[N, C_in, H, W]` | rank-four NCHW input |
| weight | `[C_out, C_in/groups, K_h, K_w]` | one stored kernel per output channel and input-channel position within its group |
| optional bias | `[C_out]` | one value added to each spatial result for its output channel |
| result | `[N, C_out, H_out, W_out]` | exact input batch and weight output-channel Dimensions plus derived spatial Dimensions |

`Conv2dAttrs` carries only intrinsic geometry and grouping. Kernel extents come from the weight
Shape, and bias presence comes from selecting the three-input overload.

| Attribute | Constraint | Meaning |
|---|---|---|
| `strideHeight`, `strideWidth` | positive `long` | step between adjacent output positions |
| `paddingHeight`, `paddingWidth` | non-negative `long` | conceptual positive-zero positions on each side of the corresponding axis |
| `dilationHeight`, `dilationWidth` | positive `long` | spacing between adjacent stored kernel positions |
| `groups` | positive `long` | number of independent contiguous input/output channel groups |

`Conv2dAttrs.defaults()` selects unit stride, zero symmetric padding, unit dilation, and one
group. The value owns no Tensor, Shape, data type, layout, storage, gradient, compiler, backend,
or execution state.

For either spatial axis, let `D` be its input extent, `k` its positive static kernel extent, `p`
the padding on each side, `d` the dilation, and `s` the stride:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = D + 2 * p - effectiveKernel
output          = floor(numerator / s) + 1
```

All literal arithmetic is checked in signed `long`. A static negative numerator fails locally;
zero produces one output position. A dynamic input extent retains the canonical symbolic
equivalent and defers the obligation that the bound numerator be non-negative. Kernel height and
width must themselves be statically known and positive.

Static input and output channel counts must each be divisible by `groups`. When the relevant
Dimensions are static, `weightChannelsPerGroup * groups` must equal `C_in`, and bias length must
equal `C_out`. A relation involving an unresolved Dimension is retained as an obligation for
current package-private compiler verification or later concrete binding because the exact
descriptors and attributes remain in provenance.

<a id="grouped-shape-and-provenance-example"></a>
#### Grouped Shape and provenance example

The goal is to inspect current metadata for a two-group convolution. Input `[2, 4, 7, 7]`, weight
`[6, 2, 3, 3]`, and bias `[6]` use two input channels per group. Unit stride and dilation with one
padding position per side preserve the `7 x 7` spatial Shape.

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

The result type is FLOAT64 because promotion processes input and weight first, then bias. The
example proves Shape, metadata, and ordered provenance only; it does not calculate convolution
values, capture a graph, construct a gradient, or establish compiler/backend support.

<a id="dynamic-height-example"></a>
#### Dynamic-height example

For input Shape `[N, 4, H, 9]`, weight Shape `[6, 4, 3, 3]`, vertical stride two, vertical padding
one, vertical dilation two, and unit horizontal geometry, height becomes:

```text
effectiveKernelHeight = 2 * (3 - 1) + 1 = 5
H_out = floor((H + 2 - 5) / 2) + 1
      = floor((H - 3) / 2) + 1
```

The result Shape is `[N, 6, floor((H - 3) / 2) + 1, 7]`. This is an exact symbolic Shape
expression, not a value chosen for `H`. A later binding must reject a concrete `H` for which the
numerator is negative.

<a id="numerical-and-metadata-policy"></a>
#### Numerical and metadata policy

For output channel `o`, its group selects the corresponding contiguous `C_in/groups` input
channels. Each result element is the optional bias for `o`, applied exactly once, plus products
over those channels and kernel height/width positions in increasing logical-index meaning.
Coordinates outside the input are conceptual positive zero and still participate in ordinary
IEEE-754 multiplication. In particular, a padded zero multiplied by an infinite weight follows
ordinary floating behavior rather than being skipped.

Input, weight, and present bias must be floating. Their promoted type is the output type. FLOAT64
output accumulates in FLOAT64. All four low types and FLOAT32 output accumulate in FLOAT32, with one
final conversion for a low output. NaN, infinity, and signed zero follow ordinary multiplication and
addition. An empty channel contraction starts from positive zero before optional bias. Empty
batch and output-channel axes remain valid. Implementations may reassociate terms and use fused
multiply-add, so traversal order, bitwise equality, and identical cross-backend rounding are not
promised.

For FLOAT32, exact CONV2D geometry, conceptual padding, bias, and contributors remain
fixed. Every term participates exactly once under the primitive and aggregate floors. Convolution
does not gain MATMUL's final-publication exact-zero choice or a final-output tolerance.

Every successful call returns one fresh, unlabeled, storage-free Tensor with unresolved layout,
the exact derived Shape, and gradient eligibility equal to the logical OR of the actual inputs.
Its one-output provenance has index zero, retains the exact `Conv2dAttrs` reference, and records
ordered inputs `[input, weight]` or `[input, weight, bias]`. Current construction reads no values
and creates no gradient rule in the Model. Package-private compiler capture, descriptor-visible
proof or retention of deferred constraints, and compiler-owned first-order input, weight, and
optional bias cotangent construction are current. The compiler formula preserves group isolation
through public unfold, matrix, reduction, and overlap-accumulating fold expressions. Legal
decomposition, concrete binding, saved values, backend lowering, and execution remain future
lifecycle work.

<a id="grouped-ncdhw-conv3d-expressions"></a>
### Grouped NCDHW Conv3d expressions

`input.conv3d(weight, attrs)` and `input.conv3d(weight, bias, attrs)` construct one first-class
grouped three-dimensional cross-correlation expression. NCDHW names the rank-five logical axis
order batch (`N`), input channel (`C_in`), depth (`D`), height (`H`), then width (`W`).
Cross-correlation uses the stored kernel in depth/height/width order without reversing it.

| Value | Required Shape | Meaning |
|---|---|---|
| receiver input | `[N, C_in, D, H, W]` | rank-five NCDHW input |
| weight | `[C_out, C_in/groups, K_d, K_h, K_w]` | one stored three-dimensional kernel per output channel and input-channel position within its group |
| optional bias | `[C_out]` | one value participating once for each spatial result in that output channel |
| result | `[N, C_out, D_out, H_out, W_out]` | exact input batch and weight output-channel Dimensions plus three derived spatial Dimensions |

`Conv3dAttrs` is an immutable ten-field value. Its fields remain in depth, height, width order
within each geometry family:

| Fields | Constraint | Meaning |
|---|---|---|
| `strideDepth`, `strideHeight`, `strideWidth` | positive `long` values | steps between adjacent output positions |
| `paddingDepth`, `paddingHeight`, `paddingWidth` | non-negative `long` values | conceptual positive-zero positions on each side of the corresponding axis |
| `dilationDepth`, `dilationHeight`, `dilationWidth` | positive `long` values | spacing between adjacent stored kernel positions |
| `groups` | positive `long` | number of independent contiguous input/output channel groups |

`Conv3dAttrs.defaults()` selects unit stride and dilation, zero symmetric padding, and one group.
Kernel extents come from the weight Shape, while choosing the three-input overload supplies bias.

For each spatial axis, let `X` be the input extent, `K_x` the positive static kernel extent, `p_x`
the symmetric padding on each side, `d_x` the dilation, and `s_x` the stride:

```text
effectiveKernel_x = d_x * (K_x - 1) + 1
numerator_x       = X + 2 * p_x - effectiveKernel_x
X_out             = floor(numerator_x / s_x) + 1
```

Literal arithmetic is checked in signed `long`. A static negative numerator fails during Model
construction, while zero produces one output position. If an input spatial Dimension is
unresolved, the result retains the canonical symbolic equivalent of the formula; its eventual
concrete value must still make the numerator non-negative. Kernel depth, height, and width must
each be statically known and positive.

Static input and output channel counts must each be divisible by `groups`. When both participating
Dimensions are static, `weightChannelsPerGroup * groups` must equal `C_in`, and bias length must
equal `C_out`. A relation involving an unresolved Dimension is deferred rather than guessed. The
exact inputs, descriptors, and attributes preserve the facts needed for later proof.

<a id="biased-grouped-conv3d-metadata-example"></a>
#### Biased grouped Conv3d metadata example

The goal is to inspect current Model metadata for two groups. Input `[2, 4, 8, 9, 10]`, weight
`[6, 2, 3, 3, 3]`, and bias `[6]` use two input channels per group. Strides `(2, 1, 2)`, symmetric
padding `(1, 1, 0)`, and dilation `(1, 2, 1)` produce spatial extents `(4, 7, 4)`.

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

The Shape follows the three formulas independently. Promotion processes input and weight first,
then bias, so the result type is FLOAT64. The last three lines identify one `CONV3D` occurrence,
the exact attributes reference, and exact ordered inputs `[input, weight, bias]`. This example
establishes Model metadata and provenance only.

<a id="numerical-and-provenance-policy"></a>
#### Numerical and provenance policy

For output channel `o`, its group selects the corresponding contiguous `C_in/groups` input
channels. Products are conceptually visited by increasing input channel, kernel depth, kernel
height, then kernel width. Optional bias participates exactly once in the accumulation domain
before final output conversion. Coordinates outside the input are conceptual positive-zero
samples and participate in ordinary IEEE-754 multiplication, including multiplication by
infinity.

Input, weight, and present bias must be floating. Promotion processes input and weight first,
then present bias. FLOAT64 output accumulates in FLOAT64; all four low types and FLOAT32 output
accumulate in FLOAT32, with one final conversion for a low output. NaN, infinity, and signed zero otherwise follow
ordinary multiplication and addition. An empty input-channel contraction starts at positive
zero before optional bias; empty batch and output-channel axes are valid. Reassociation and fused
multiply-add are permitted, so no fixed summation order or bitwise-identical rounding is promised.

For FLOAT32, exact CONV3D geometry, conceptual padding, bias, and contributors remain
fixed. Every term participates exactly once under the primitive and aggregate floors. Convolution
does not gain MATMUL's final-publication exact-zero choice or a final-output tolerance.

Each successful call creates one fresh canonical output at producer index zero. The result is
unlabeled and storage-free, has unresolved layout, retains the exact derived Shape and promoted
type, and combines the input gradient-request metadata with logical OR. Provenance retains the
exact `Conv3dAttrs` reference and ordered inputs `[input, weight]` or
`[input, weight, bias]`. Current Model construction reads no tensor values. Compiler independently
infers and final-validates these descriptors and deferred channel/spatial relations. Its shared
Conv2d/Conv3d first-order rule retains groups through unfold/MATMUL/fold/reductions, restores
each selected input/weight/bias target's exact Shape/type, and accumulates overlapping input
contributions through the public fold. Actual execution is qualified for static CPU FLOAT32/FLOAT64
and Metal FLOAT32, not ordinary dynamic/named-B Conv3d, low CPU arithmetic or arbitrary layouts;
see the [production training boundary](../../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary).


---

[Previous topic](attention.md) · [Next topic](pooling.md)
