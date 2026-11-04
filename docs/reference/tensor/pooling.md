# Pool1d, Pool2d and Pool3d

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**Reader model.** Pooling summarizes a moving neighborhood independently in each channel.
Maximum pooling selects its greatest eligible sample; average pooling adds samples and divides
by a count. Unlike convolution, pooling has no learned kernel weights and does not combine
channels. Window **stride** moves the neighborhood between output positions; **dilation**
spaces samples inside it. **Padding** describes positions outside the original domain, but its
treatment differs by pooling family: maximum excludes them, while this average-pooling contract
counts their positive-zero contributions. Floor and ceiling modes round the output-grid quotient
down or up; they do not round input values. The existing Pool1d diagram shows its singleton-height
composition, and the later numerical examples explain border windows.

<a id="ncw-pool1d-composition"></a>
### NCW Pool1d composition

`input.maxPool1d(attrs)` and `input.averagePool1d(attrs)` are rank-specific conveniences for
one-dimensional pooling in NCW order: batch, channel, width. They do not add maximum- or
average-Pool1d operation kinds. Each successful call creates exactly this visible composition:

```text
[N, C, W] --EXPAND_DIMS(axis 2)--> [N, C, 1, W]
          --MAX_POOL2D or AVERAGE_POOL2D--> [N, C, 1, W_out]
          --SQUEEZE(axis 2)--> [N, C, W_out]
```

`MaxPool1dAttrs` and `AveragePool1dAttrs` are immutable public composition parameters. Both carry
positive `kernelWidth`, `strideWidth`, and `dilationWidth`, non-negative symmetric
`paddingWidth`, and literal `ceilMode`. Construction maps those components to a fresh matching
Pool2d attributes value:

```text
height: kernel=1,           stride=1,           padding=0,            dilation=1
width:  kernel=kernelWidth, stride=strideWidth, padding=paddingWidth, dilation=dilationWidth
```

The singleton height has output extent one in both floor and literal-ceil modes. Width therefore
uses the existing Pool2d formula without alteration:

```text
effectiveKernel = dilationWidth * (kernelWidth - 1) + 1
numerator       = W + 2 * paddingWidth - effectiveKernel
W_out           = floor(numerator / strideWidth) + 1
```

When `ceilMode` is true, the quotient uses literal ceiling division. A static negative numerator
fails before any producer is created. Checked `long` overflow fails with `ArithmeticException`.
An unresolved width retains the exact symbolic floor- or ceiling-division expression and the
obligation that a later concrete binding make the numerator non-negative. The receiver must be a
floating rank-three Tensor; null attributes, a non-floating type, and another rank fail locally.

Maximum pooling inherits Pool2d padding exclusion and increasing logical width order. An
in-bounds NaN is dominant at its first logical occurrence, positive zero ranks above negative
zero, equal values retain their first occurrence, and an all-padding window produces negative
infinity. Average pooling inherits the fixed count-padding divisor, which reduces from
`1 * kernelWidth` to `kernelWidth`; padding contributes positive zero while still counting.
All four low types and FLOAT32 average accumulation and division use FLOAT32; FLOAT64 uses FLOAT64,
and a low result narrows once. The existing Pool2d NaN, infinity, signed-zero, rounding,
reassociation, and all-padding policies otherwise apply unchanged.

<a id="pool1d-shape-and-provenance-example"></a>
#### Pool1d Shape and provenance example

The goal is to inspect maximum and average Pool1d metadata for input `[2, 3, 9]`. A width-three
kernel, stride two, one padding position per side, unit dilation, and floor mode produce width
five.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool1dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool1dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

Tensor input = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(2, 3, 9), Optional.empty(), true));
Tensor maximum = input.maxPool1d(new MaxPool1dAttrs(3, 2, 1, 1, false));
Tensor average = input.averagePool1d(new AveragePool1dAttrs(3, 2, 1, 1, false));

var maximumSqueeze = maximum.provenance().orElseThrow();
var maximumPool2d = maximumSqueeze.inputs().getFirst().provenance().orElseThrow();
var maximumExpand = maximumPool2d.inputs().getFirst().provenance().orElseThrow();

System.out.println(maximum.descriptor().shape());
System.out.println(average.descriptor().shape());
System.out.println(maximumExpand.operation().kind());
System.out.println(maximumPool2d.operation().kind());
System.out.println(maximumSqueeze.operation().kind());
```

It prints:

```text
Shape[2, 3, 5]
Shape[2, 3, 5]
EXPAND_DIMS
MAX_POOL2D
SQUEEZE
```

Each call returns a fresh canonical, unlabeled, storage-free Tensor with unresolved layout. It
retains the input data type, gradient request, and exact batch/channel Dimension references while
recording three fresh one-output producers in order. Compiler capture, descriptor validation, and
gradients consequently see the ordinary rank-edit and Pool2d occurrences. Maximum gradients
inherit Pool2d's reconstructed first eligible winner; average gradients inherit its fixed divisor
and overlap fold. There is no Pool1d signature, captured node, dedicated gradient rule, or saved
maximum-index output.

Backend support is the conjunction of support for the exact `EXPAND_DIMS`, matching Pool2d, and
`SQUEEZE` occurrences with their actual descriptors and attributes. A backend may recognize the
whole topology as an optimization only if it preserves that meaning; neither this Model API nor
such recognition creates or advertises a Pool1d operation capability. The current CPU backend
recognizes only this exact private single-use topology with resolved singleton-height affine
layouts. It keeps both rank edits virtual and reuses the Pool2d family projection under the
current schema-70 envelope
bytes that the matching direct Pool2d occurrence uses. Near matches retain ordinary decomposition.
The example itself still proves Model metadata and provenance only: constructing it does not read
values, execute pooling, guarantee that CPU recognition will apply, or promise fusion.

<a id="nchw-maximum-pooling-expressions"></a>
### NCHW maximum-pooling expressions

`input.maxPool2d(attrs)` constructs one `MAX_POOL2D` expression over a floating rank-four input in
NCHW axis order: batch (`N`), channel (`C`), height (`H`), then width (`W`). Pooling moves a
two-dimensional window over each batch/channel plane and requests the greatest in-bounds sampled
value. Current construction records that meaning and derives metadata; it does not read input
values or perform pooling.

The input and result contracts are:

| Value | Required Shape and type | Result meaning |
|---|---|---|
| receiver input | `[N, C, H, W]`; floating, including FP8 | one rank-four NCHW source |
| result | `[N, C, H_out, W_out]`; exact input type | exact input `N` and `C` Dimension references plus derived spatial Dimensions |

`MaxPool2dAttrs` carries the complete intrinsic window geometry. Each successful call retains the
exact attributes reference in provenance.

| Attribute | Constraint | Meaning |
|---|---|---|
| `kernelHeight`, `kernelWidth` | positive `long` | number of logical samples in the corresponding kernel axis |
| `strideHeight`, `strideWidth` | positive `long` | distance between consecutive window starts |
| `paddingHeight`, `paddingWidth` | non-negative `long` | symmetric coordinate positions per side; these positions are excluded from maximum selection |
| `dilationHeight`, `dilationWidth` | positive `long` | spacing between consecutive kernel samples |
| `ceilMode` | boolean | use the literal ceiling grid when true and the floor grid when false |

For either spatial axis, let `D` be its input extent, `k` the kernel sample count, `p` the
padding on each side, `d` the dilation, and `s` the stride:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = D + 2 * p - effectiveKernel
floor output    = floor(numerator / s) + 1
ceil output     = ceil(numerator / s) + 1
```

All literal geometry uses checked signed-`long` arithmetic. A static negative numerator fails
before result construction. For an unresolved extent, construction retains the canonical
`addConstant`, floor-or-ceiling divide, then `addConstant` expression. A later concrete binding
must satisfy the retained obligation that the numerator is non-negative.

<a id="static-floor-and-literal-ceil-example"></a>
#### Static floor and literal-ceil example

The goal is to compare floor and ceil grids for input Shape `[1, 1, 2, 2]` with a one-sample
kernel, stride three, no padding, and unit dilation.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class MaxPool2dShapeExample {
    public static void main(String[] args) {
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(1, 1, 2, 2), Optional.empty(), true));
        MaxPool2dAttrs floor = new MaxPool2dAttrs(1, 1, 3, 3, 0, 0, 1, 1, false);
        MaxPool2dAttrs ceil = new MaxPool2dAttrs(1, 1, 3, 3, 0, 0, 1, 1, true);

        System.out.println(input.maxPool2d(floor).descriptor().shape());
        System.out.println(input.maxPool2d(ceil).descriptor().shape());
    }
}
```

It prints:

```text
Shape[1, 1, 1, 1]
Shape[1, 1, 2, 2]
```

For each axis, the numerator is `2 - 1 = 1`. Floor division gives zero, then adding one gives
one output position. Ceiling division gives one, then adding one gives two. The second position
starts beyond the input, but the selected literal ceil grid does not apply a terminal-window
decrement. Its all-padding window has the eventual value negative infinity. This example proves
Shape metadata only; current model code does not calculate either output Tensor's values.

<a id="symbolic-floor-and-ceil-example"></a>
#### Symbolic floor and ceil example

For input Shape `[N, C, H, W]`, kernel `3 x 5`, stride `2 x 3`, padding `1 x 0`, and dilation
`2 x 1`, the effective kernels are five in both axes. Construction retains:

```text
floor Shape = [N, C, floor((H - 3) / 2) + 1, floor((W - 5) / 3) + 1]
ceil Shape  = [N, C, ceil((H - 3) / 2) + 1,  ceil((W - 5) / 3) + 1]
```

These are exact symbolic expressions, not estimated extents. A later binding must reject any
concrete `H` below three or `W` below five because that would make a numerator negative.

<a id="extrema-and-padding-policy"></a>
#### Extrema and padding policy

For output coordinate `o`, a spatial window starts at `o * stride - padding`. Logical kernel
sample `r` reads coordinate `start + r * dilation`. Samples outside the input are excluded rather
than converted to zero. If no sampled coordinate is in bounds, the result is exact negative
infinity in the input type. Otherwise selection follows these rules:

- any not-a-number (NaN) candidate wins over every non-NaN, so NaN propagates;
- multiple NaNs select the first logical kernel sample, although NaN payload/sign and signaling
  preservation are unspecified;
- positive zero wins over negative zero;
- infinities use ordinary numerical order; and
- other equal candidates select the first sample in increasing kernel-height, then kernel-width,
  order.

Every success returns one fresh, unlabeled, storage-free Tensor with unresolved layout. It
retains the exact input type, `N` and `C` Dimension references, and `requiresGrad` request. Its
one-output provenance has index zero and ordered inputs `[input]`. The gradient request is
Model metadata, not a Model-owned gradient implementation. Package-private compiler capture,
descriptor-visible proof or retention of deferred constraints, and compiler-owned first-order
input-cotangent construction are current. The formula reconstructs the exact first eligible
logical winner from the original input and same-occurrence public output, including padding,
NaN, and signed-zero policy; it adds no saved-index output. Qualified forward and generated
backward occurrences already lower, prepare and execute through backend owners. Construction
itself does not perform these stages or grant arbitrary dynamic pooling and decomposition.

<a id="nchw-average-pooling-expressions"></a>
### NCHW average-pooling expressions

`input.averagePool2d(attrs)` is the canonical average-pooling receiver expression. It constructs
one `AVERAGE_POOL2D` occurrence for a floating rank-four NCHW input. Each output combines one
logical two-dimensional window using a fixed count-padding divisor: every kernel position counts,
including positions outside the input. Current construction records this meaning and derives
metadata; it does not read values or calculate an average.

| Value | Required Shape and type | Result meaning |
|---|---|---|
| receiver input | `[N, C, H, W]`; floating, including FP8 | one rank-four NCHW source |
| result | `[N, C, H_out, W_out]`; exact input type | exact input `N` and `C` Dimension references plus derived spatial Dimensions |

`AveragePool2dAttrs` is distinct from `MaxPool2dAttrs` because average pooling owns divisor and
accumulation semantics rather than extrema semantics. Each successful call retains the exact
attributes reference in provenance.

| Attribute or policy | Constraint | Meaning |
|---|---|---|
| `kernelHeight`, `kernelWidth` | positive `long` | logical sample counts; their positive mathematical product is the fixed divisor |
| `strideHeight`, `strideWidth` | positive `long` | distance between consecutive window starts |
| `paddingHeight`, `paddingWidth` | non-negative `long` | symmetric positions per side; each contributes conceptual positive zero and still counts in the divisor |
| `dilationHeight`, `dilationWidth` | positive `long` | spacing between consecutive kernel positions; it does not change the divisor |
| `ceilMode` | boolean | use the literal ceiling grid when true and the floor grid when false |
| accumulation and division | fixed by input type | All four low types and FLOAT32 use FLOAT32; FLOAT64 uses FLOAT64; divide the accumulated sum once |

There is no count-padding option, valid-sample averaging mode, or divisor override. For either
spatial axis, input extent `D`, kernel-position count `k`, padding per side `p`, dilation `d`, and
stride `s` produce:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = D + 2 * p - effectiveKernel
floor output    = floor(numerator / s) + 1
ceil output     = ceil(numerator / s) + 1
```

Literal geometry uses checked signed-`long` arithmetic. A static negative numerator fails before
result construction. An unresolved extent retains the canonical add, floor-or-ceiling divide,
then add expression plus the obligation that a later concrete binding makes the numerator
non-negative.

<a id="static-floor-literal-ceil-and-symbolic-examples"></a>
#### Static floor, literal-ceil, and symbolic examples

For input Shape `[1, 1, 2, 2]`, a `1 x 1` kernel, `3 x 3` stride, zero padding, and unit dilation,
floor mode produces Shape `[1, 1, 1, 1]`, while literal ceil mode produces
`[1, 1, 2, 2]`. The terminal ceil-grid positions begin in trailing padding and therefore describe
all-padding windows; average pooling retains them and their eventual value is positive zero.

For symbolic input Shape `[N, C, H, W]`, kernel `3 x 5`, stride `2 x 3`, padding `1 x 0`, and
dilation `2 x 1`, both effective kernels are five. Construction retains these exact expressions:

```text
floor Shape = [N, C, floor((H - 3) / 2) + 1, floor((W - 5) / 3) + 1]
ceil Shape  = [N, C, ceil((H - 3) / 2) + 1,  ceil((W - 5) / 3) + 1]
```

These examples prove current Shape metadata only. They do not bind `H` or `W`, evaluate a window,
or establish compiler or backend support.

<a id="count-padding-numerical-example"></a>
#### Count-padding numerical example

Consider one conceptual FLOAT32 input plane with values `[[1, 2], [3, 4]]`, a `2 x 2` kernel,
unit stride and dilation, and padding one. The top-left window contains input value `1` and three
padding positions. The selected average is:

```text
count-padding numerator = 1 + 0 + 0 + 0 = 1
fixed divisor           = 2 * 2 = 4
result                  = 1 / 4 = 0.25
```

Valid-sample averaging would divide by one and return `1.0`; that is deliberately not this
operation. The calculation explains eventual semantics. Current model construction does not
produce `0.25` or allocate result storage.

<a id="numerical-empty-and-metadata-policy"></a>
#### Numerical, empty, and metadata policy

Every in-bounds value contributes once to the numerator and every out-of-bounds position
contributes exact positive zero. All four low types and FLOAT32 accumulate and divide in FLOAT32;
FLOAT64 uses FLOAT64. A low result narrows once after division. The accumulator-domain sum is
divided exactly once by `kernelHeight * kernelWidth`.

- Any in-bounds not-a-number (NaN) makes the result NaN.
- Positive and negative infinity together make the result NaN; otherwise an infinity retains its
  sign.
- An exact-zero finite mean is negative zero only when every divisor contribution is an in-bounds
  negative zero. Cancellation, any positive zero, or any padding contribution produces positive
  zero.
- An all-padding window therefore returns exact positive zero.
- Empty batch or channel extents are valid and produce no values. Positive kernel counts mean
  there is no zero-divisor case.

The aggregate floor permits rounded summation trees while retaining every mapped contributor
and the fixed divisor. Arithmetic sites obey their declared working dtype, rounding and DAZ/FTZ
rules; no finite tolerance changes membership, padding or special-value requirements.
Backend/route finite tolerances are test-only, not a later production policy. Traversal order,
NaN payload/sign preservation and cross-backend bitwise identity are not promised.

Every success returns one fresh, unlabeled, storage-free Tensor with unresolved layout. It retains
the exact input type, `N` and `C` Dimension references, and `requiresGrad` request. Its one-output
provenance has index zero and ordered inputs `[input]`. The gradient request is metadata only.
Package-private compiler capture and descriptor-visible proof or retention of deferred constraints
are current, as is compiler-owned first-order input-cotangent construction through a logical typed
fixed kernel count and public expansion/fold expressions. Qualified occurrences already have
backend lowering, preparation and execution. Their derivative topology must qualify in full,
preserving the fixed divisor and special-value policy; construction adds no implicit route
or permission for an arbitrary dynamic decomposition.

<a id="ncdhw-pool3d-expressions"></a>
### NCDHW Pool3d expressions

`input.maxPool3d(maxAttrs)` and `input.averagePool3d(averageAttrs)` construct one first-class
three-dimensional pooling occurrence over a floating rank-five tensor in NCDHW order: batch
(`N`), channel (`C`), depth (`D`), height (`H`), then width (`W`). Model construction records
geometry, numerical meaning, and provenance only. It does not read input values or produce a
materialized numerical result.

| Value | Required Shape and type | Result meaning |
|---|---|---|
| receiver input | `[N, C, D, H, W]`; floating, including FP8 | one rank-five NCDHW source |
| result | `[N, C, D_out, H_out, W_out]`; exact input type | exact input `N` and `C` Dimension references plus three derived spatial Dimensions |

`MaxPool3dAttrs` and `AveragePool3dAttrs` are separate immutable thirteen-component records. In
each record, kernel, stride, padding, and dilation fields use depth, height, then width order;
`ceilMode` follows those twelve values. Kernel, stride, and dilation components are positive.
Symmetric padding per side is non-negative. There is no geometry array, asymmetric intrinsic
padding, adaptive/global mode, valid-sample average, or divisor override.

For each spatial input extent `X`, kernel count `k`, symmetric padding per side `p`, dilation `d`,
and stride `s`:

```text
effectiveKernel = d * (k - 1) + 1
numerator       = X + 2 * p - effectiveKernel
floor output    = floor(numerator / s) + 1
ceil output     = ceil(numerator / s) + 1
```

`ceilMode == true` selects the literal ceiling formula independently for depth, height, and
width. It does not remove a terminal window that begins wholly in trailing padding. Static
arithmetic is checked in signed `long`, and a static negative numerator fails before producer
creation. An unresolved extent retains the canonical add, matching floor-or-ceiling divide, and
final add expression; a later concrete binding must still make its numerator non-negative.
Validation and spatial derivation proceed depth, then height, then width.

Maximum pooling excludes out-of-bounds positions from selection. Eligible logical kernel
coordinates are ordered by increasing depth, then height, then width. An in-bounds not-a-number
(NaN) dominates every non-NaN and the first eligible NaN wins. Otherwise ordinary maximum
applies, positive zero ranks above negative zero, infinities use ordinary order, and equal
candidates retain the first eligible occurrence. A window with no in-bounds coordinate has exact
negative infinity in the input type. A selected non-NaN value keeps its represented input value;
there is no accumulator or saved-index output.

Average pooling uses the fixed positive mathematical divisor
`kernelDepth * kernelHeight * kernelWidth`. Every logical kernel position counts. An in-bounds
coordinate contributes its value once; an out-of-bounds coordinate contributes conceptual
positive zero and still counts. Dilation changes coordinates, not the divisor, and Model
construction does not materialize the three-factor product as a `long`. BFLOAT16, FLOAT16, and
FLOAT32 accumulate and divide in FLOAT32; FLOAT64 uses FLOAT64, the sum is divided once, and a low
result narrows once after division. Finite summation may be reassociated.

For average pooling, an in-bounds NaN produces NaN. Opposing infinity signs produce NaN;
otherwise an infinity retains its sign. An exact-zero finite mean is negative zero only when
every divisor position is an in-bounds negative zero. Cancellation, any positive zero, or any
padding contribution produces positive zero, so an all-padding window produces positive zero.
No traversal, bitwise cross-backend result, or NaN representation is promised.

<a id="pool3d-metadata-example"></a>
#### Pool3d metadata example

The goal is to inspect metadata for input Shape `[1, 2, 4, 5, 6]`. A `2 x 3 x 2` kernel,
stride `(2, 2, 3)`, symmetric padding `(1, 1, 1)`, and unit dilation produce output Shape
`[1, 2, 3, 3, 3]` in floor mode.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

Tensor input = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(1, 2, 4, 5, 6), Optional.empty(), true));
MaxPool3dAttrs maxAttrs = new MaxPool3dAttrs(
        2, 3, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1, false);
AveragePool3dAttrs averageAttrs = new AveragePool3dAttrs(
        2, 3, 2, 2, 2, 3, 1, 1, 1, 1, 1, 1, false);

Tensor maximum = input.maxPool3d(maxAttrs);
Tensor average = input.averagePool3d(averageAttrs);

System.out.println(maximum.descriptor().shape());
System.out.println(average.descriptor().shape());
System.out.println(maximum.provenance().orElseThrow().operation().kind());
System.out.println(average.provenance().orElseThrow().operation().kind());
System.out.println(maximum.provenance().orElseThrow().inputs().getFirst() == input);
```

It prints:

```text
Shape[1, 2, 3, 3, 3]
Shape[1, 2, 3, 3, 3]
MAX_POOL3D
AVERAGE_POOL3D
true
```

Each call creates a fresh, unlabeled, storage-free canonical Tensor with unresolved layout. The
result retains the exact input type, batch/channel Dimension references, and gradient-request
metadata. Provenance retains the exact supplied attributes reference, ordered input `[input]`,
output index zero, and the canonical `producer.output(0)` wrapper. The example proves those
current Model facts only. Complete Compiler tasks 0006B1 and 0006B2 capture, validate, and
differentiate both Pool3d signatures through ordinary public Tensor expressions. Average Pool3d
uses its fixed logical kernel divisor and overlap-accumulating `fold3d`. Maximum Pool3d uses the
exact same-occurrence output and an explicit in-bounds eligibility mask so dominant NaN,
signed-zero ordering, real negative infinity, first-winner, and all-padding behavior remain
distinguishable; its selected mask is fixed for a later derivative stage. Current Model owns the
general `unfold3d`/`fold3d` algebra. CPU FLOAT32/FLOAT64 now admit matching input/output gradient
flags and actual Pool3d backward closure through these existing shared rules and window families.
Direct forward execution retains fully static Shapes, resolved non-negative layouts, injective
output layouts and checked geometry. CPU low pooling forward and selected actual gradients use
the same owners with FLOAT32 arithmetic and final narrowing. Existing
scalar/caller-parallel forward emission owns complete output cells with zero workspace or
materialization; no standalone backward kernel or vector/native/fused pooling route was added.
Qualified CPU all-valid/runtime-length Scan composes both pools and their actual reverse bodies
in homogeneous FLOAT32/FLOAT64 and both traversal directions, under existing complete-body,
static-window and terminal restrictions. Ordinary public named-B admission is unchanged.
Metal FLOAT32 admits overlapping Pool2d/Pool3d gradient-bearing windows through the same
closure; original-carrier Java/raw-native checks retain FLOAT64/low non-overlap restrictions.
No low arithmetic expansion, Metal Scan or fallback follows. These are locally runtime-qualified
capabilities, not joined-release acceptance; construction itself still performs no evaluation.


---

[Previous topic](convolution.md) · [Next topic](sort-topk.md)
