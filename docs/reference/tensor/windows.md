# Window transforms

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="window-transform-expressions"></a>
### Window-transform expressions

The current public API constructs eight storage-free sliding-window expressions. The public
`Tensor` class has exactly 213 declared public methods after adding the three rank-specific 3D
receivers:

```java
tensor.unfold(int axis, long size, long step)
tensor.foldAxis(int axis, long outputSize, long step)
tensor.unfold2d(Window2dAttrs window)
tensor.unfold2d(Window2dAttrs window, ScalarValue paddingValue)
tensor.fold2d(Shape outputShape, Window2dAttrs window)
tensor.unfold3d(Window3dAttrs window)
tensor.unfold3d(Window3dAttrs window, ScalarValue paddingValue)
tensor.fold3d(Shape outputShape, Window3dAttrs window)
```

`unfold` requires rank at least one and normalizes its raw axis against the input rank. The selected
dimension must be static, `size` and `step` must be positive, and `size` must fit that dimension.
For selected extent `D`, it replaces that extent with
`floor((D - size) / step) + 1`, preserves every unaffected exact Dimension reference, and appends
`size` as the final axis. It accepts every current data type. Thus input Shape `[2, 5, 3]`, axis
`1`, size `3`, and step `1` produces `[2, 3, 3, 3]`.

`foldAxis` reverses that rank change under overlap summation. Its final input Dimension is a
positive static window size; its remaining Dimensions define the target rank used to normalize
the raw axis. The selected static window-count Dimension is replaced by non-negative
`outputSize`, and the final Dimension is removed. Positive output size requires
`floor((outputSize - windowSize) / step) + 1` windows; zero output size requires zero windows.
Floating and signed-integral inputs are accepted, while BOOL is rejected. For conceptual windows
`[[1,2,3], [4,5,6], [7,8,9]]`, axis `0`, output size `5`, and step `1`, overlap summation gives
conceptual result `[1,6,15,14,9]`.

Both `unfold2d` overloads require rank-four NCHW input and a floating data type. Batch, channel,
height, and width may be static or unresolved; the exact batch Dimension is retained. `fold2d`
requires floating rank-three canonical columns and an explicit rank-four NCHW output Shape.
Column and output batch Dimensions must be structurally equal. For each spatial dimension, all
three 2D construction paths use checked static arithmetic or retain the corresponding canonical
symbolic expression:

```text
effectiveKernel = dilation * (kernel - 1) + 1
numerator       = input + 2 * padding - effectiveKernel
positions       = floor(numerator / stride) + 1       in floor mode
positions       = ceil(numerator / stride) + 1        in ceil mode
```

The effective kernel must fit the padded input when static geometry can prove otherwise. An
unresolved numerator retains its non-negativity obligation for later binding. Ceil mode avoids
the overflow-prone `numerator + stride - 1` identity. `unfold2d` produces canonical Shape
`[N, C * kernelHeight * kernelWidth, outputHeight * outputWidth]`; products of unresolved
Dimensions use the exact canonical symbolic-product form. `fold2d` requires complete structural
equality with those channel and window-count formulas and retains the exact supplied output Shape.
An unrelated unresolved symbol is rejected because model construction does not create an unnamed
equality constraint.

For dynamic input `[N,C,H,W]`, a 3-by-3 kernel, stride two, padding one, unit dilation, and floor
mode produces channel Dimension `9 * C` and flattened window Dimension
`(floor((H - 1) / 2) + 1) * (floor((W - 1) / 2) + 1)`. Ceil mode retains the same formula with
ceiling division. This metadata is exact structurally; it does not bind `N`, `C`, `H`, or `W`.

The one-argument `unfold2d` pairs `UNFOLD2D` directly with the supplied `Window2dAttrs`; every
out-of-domain sample is conceptual positive zero. The two-argument overload instead pairs
`UNFOLD2D` with `Unfold2dAttrs(window, paddingValue)`. The scalar must exactly match the floating
input data type and is retained by reference with its raw NaN, infinity, or signed-zero bits. It
supplies symmetric padding positions and terminal ceil-grid positions beyond the padded extent.
This is extraction semantics, not maximum-pooling padding or a backend fill policy.

Conceptual `[1, 1, 3, 3]` input with a 2-by-2 kernel, unit stride and dilation, zero symmetric
padding, and floor mode unfolds to `[1, 4, 4]`. Folding compatible columns to `[1, 1, 3, 3]`
scatter-adds four contributions at the center and one at each corner, without overlap averaging.

The 3D forms extend the same algebra to rank-five NCDHW input `[N, C, D, H, W]`. They accept
every floating type, including FP8. For each depth, height, and width axis they use the same
literal floor-or-ceiling formula above, deriving depth first, then height, then width. They do not
drop a terminal literal-ceil window even when every sampled coordinate lies outside the unpadded
input. `unfold3d` always returns the canonical rank-three Shape:

```text
[N, C * kD * kH * kW, DOut * HOut * WOut]
```

For output-grid coordinate `(od, oh, ow)` and kernel coordinate `(kd, kh, kw)`, the logical input
coordinate and two flattened column coordinates are:

```text
id = od * strideDepth  - paddingDepth  + kd * dilationDepth
ih = oh * strideHeight - paddingHeight + kh * dilationHeight
iw = ow * strideWidth  - paddingWidth  + kw * dilationWidth

q = (((c * kD + kd) * kH + kh) * kW + kw)
p = ((od * HOut + oh) * WOut + ow)
column = [n, q, p]
```

| Flattened coordinate | Outermost to fastest-changing component | Meaning |
|---|---|---|
| `q` | channel, kernel depth, kernel height, kernel width | selects one channel/kernel position |
| `p` | output depth, output height, output width | selects one spatial window position |

This ordering is structural Model metadata. It fixes the later interpretation of columns but does
not promise a physical storage traversal or backend algorithm.

For direct `unfold3d(window)`, every out-of-range `(id, ih, iw)` sample is exact represented
positive zero. The typed overload instead retains and uses the exact same-typed `ScalarValue`, so
`ScalarValue.float32(Float.NEGATIVE_INFINITY)` can describe maximum-reconstruction padding
without adding a pooling-specific operation. The scalar's reference and represented bits are
retained; no conversion occurs.

`fold3d(outputShape, window)` requires rank-three columns whose batch, complete channel-kernel
Dimension, and complete flattened-grid Dimension are structurally equal to the formulas for the
explicit rank-five NCDHW target. The result retains the exact supplied `Shape` reference.
Out-of-range padded and literal-ceil-tail coordinates are excluded geometrically, regardless of
the column value. Each target starts at represented positive zero, and in-range contributions are
added in increasing `[n, q, p]` order. FLOAT64 and FLOAT32 add sequentially in their own format.
BFLOAT16 expands the accumulator and next operand to FLOAT32, adds once, and narrows after every
logical contribution. Fold neither averages overlaps nor compares values with a padding scalar.

As a small overlap example, a one-channel `[1, 1, 3, 3, 3]` target with a 2-by-2-by-2 kernel,
unit stride/dilation, zero padding, and floor mode has columns Shape `[1, 8, 8]`. The center target
coordinate belongs to all eight windows, while each corner belongs to one. `fold3d` therefore
adds eight distinct column contributions at the center and one at each corner. This describes the
eventual value semantics; Model construction itself performs no addition.

For unresolved input `[N, C, D, H, W]`, a 2-by-3-by-4 kernel, stride `(2, 2, 3)`, padding
`(1, 1, 2)`, unit dilation, and floor mode produces:

```text
[N,
 ((C * 2) * 3) * 4,
 ((floor(D / 2) + 1) * (floor((H - 1) / 2) + 1)) * (floor(W / 3) + 1)]
```

The exact `N` Dimension reference is retained. These canonical expressions describe unresolved
metadata; they do not bind the symbols or discharge the later non-negative-numerator obligation.

All results preserve the exact input data type and gradient-eligibility flag, use unresolved
layout, record the matching normalized attributes and exact `[input]` provenance, and receive a
fresh unlabeled identity. They do not inspect input layout, label, provenance, storage, or values;
attach or allocate storage; materialize windows; accumulate overlaps; create a gradient rule;
capture a graph; lower to a backend; or execute.

Current Model owns `UNFOLD3D`/`FOLD3D` algebra and public metadata construction. Complete Compiler
0006B1 captures both unfold variants and fold as ordinary flat nodes and independently validates
their canonical descriptors and spatial obligations. Complete Compiler 0006B2 constructs all
three adjoints through the same public API: either unfold form maps its cotangent through
`fold3d` into the exact original input Shape, and fold maps its cotangent through the direct
positive-zero `unfold3d` overload. Typed padding is immutable configuration rather than a
differentiable input. CPU separately executes fully static resolved-layout UNFOLD3D raw movement
for every floating type, including FP8, and FOLD3D overlap arithmetic under ordinary dtype/geometry
guards. Metal FOLD2D/FOLD3D overlap is FLOAT32-only, while its other floating image folds retain
non-overlap. CPU floating and Metal F32 Pool3d training reuse the shared pooling adjoints and
these 3D windows; CPU parity does not widen Metal's independent domain.
These capabilities require concrete backend admission; constructing metadata alone supplies no
materialized result or performance guarantee. See the
[production matrix](../../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary).

<a id="complete-window-transform-expression-example"></a>
#### Complete window-transform expression example

This runnable Java 26 example observes only Shape, operation, provenance, and storage metadata:

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
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
        Tensor folded = tensor(Shape.of(3, 3)).foldAxis(0, 5, 1);
        Window2dAttrs window = new Window2dAttrs(
                2, 2, 1, 1, 0, 0, 1, 1, false);
        Tensor columns = tensor(Shape.of(1, 1, 3, 3)).unfold2d(window);
        Tensor image = columns.fold2d(Shape.of(1, 1, 3, 3), window);
        ScalarValue padding = ScalarValue.float32(-0.0f);
        Tensor paddedColumns = tensor(Shape.of(1, 1, 3, 3)).unfold2d(window, padding);

        Dimension n = new DynamicDimension("N");
        Dimension c = new DynamicDimension("C");
        Dimension h = new DynamicDimension("H");
        Dimension w = new DynamicDimension("W");
        Shape dynamicImageShape = Shape.ofDimensions(n, c, h, w);
        Window2dAttrs dynamicWindow = new Window2dAttrs(
                3, 3, 2, 2, 1, 1, 1, 1, false);
        Tensor dynamicColumns = tensor(dynamicImageShape).unfold2d(dynamicWindow);
        Tensor dynamicImage = dynamicColumns.fold2d(dynamicImageShape, dynamicWindow);
        Unfold2dAttrs paddingAttrs = (Unfold2dAttrs) paddedColumns.provenance()
                .orElseThrow().operation().attrs();

        System.out.println(unfolded.descriptor().shape());
        System.out.println(folded.descriptor().shape());
        System.out.println(columns.descriptor().shape());
        System.out.println(image.descriptor().shape());
        System.out.println(paddingAttrs.paddingValue() == padding);
        System.out.println(dynamicColumns.descriptor().shape().dimensions().get(1)
                .equals(DimensionExpressions.multiply(c, 9)));
        System.out.println(dynamicImage.descriptor().shape() == dynamicImageShape);
        System.out.println(image.descriptor().layout().isEmpty()
                && image.hostStorage().isEmpty());
    }
}
```

It prints:

```text
Shape[2, 3, 3, 3]
Shape[5]
Shape[1, 4, 4]
Shape[1, 1, 3, 3]
true
true
true
true
```

Invalid rank, axis, intrinsic sign, required staticity, window fit, type, or column compatibility
fails before identity allocation. Structural `fold2d` compatibility rejects unrelated unresolved
Dimensions. Checked geometry or canonicalization overflow propagates as `ArithmeticException`.
Null references are rejected in parameter order. Identifier exhaustion remains possible only at
final derived-Tensor construction after all local immutable metadata has been prepared.


---

[Previous topic](composition.md) · [Next topic](matmul-linear.md)
