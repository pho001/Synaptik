# Pool1d, Pool2d and [Pool3d](../../glossary.md#pool3d--ncdhw-pooling)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## A pooling call describes one summary per window

Application or NN layer-building code calls a pooling method on the input Tensor with immutable
window attributes. The method retains the exact input and attribute references, derives output
dimensions and returns a storage-free expression. It does not inspect the input's numbers,
copy them or own a result buffer yet. [Window transforms](windows.md) explain the sampled
coordinates; [convolution](convolution.md) uses similar geometry but adds learned weights and
channel mixing. Consult the [production matrix](../../backend-guide/production-support.md) for
the independent execution and generated-gradient domains.

**[Reader model](../../glossary.md#term-reader-model).** [Pooling](../../glossary.md#term-pooling) summarizes a moving neighborhood independently in each [channel](../../glossary.md#term-channel).
Maximum pooling selects its greatest eligible [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters); average pooling adds samples and divides
by a count. Unlike [convolution](../../glossary.md#term-channels-first-convolution-layers), pooling has no learned [kernel](../../glossary.md#kernel) [weights](../../glossary.md#term-weight) and does not combine
channels. Window **[stride](../../glossary.md#term-stride)** moves the neighborhood between output positions; **[dilation](../../glossary.md#term-dilation)**
spaces samples inside it. **[Padding](../../glossary.md#padding)** describes positions outside the original domain, but its
treatment differs by pooling family: maximum excludes them, while this average-pooling contract
counts their positive-zero contributions. Floor and ceiling modes round the output-grid quotient
down or up; they do not round input values. The existing Pool1d diagram shows its [singleton](../../glossary.md#term-singleton)-height
composition, and the later numerical examples explain border windows.

<a id="ncw-pool1d-composition"></a>
### [NCW](../../glossary.md#term-ncw) [Pool1d composition](../../glossary.md#pool1d-composition)

`input.maxPool1d(attrs)` and `input.averagePool1d(attrs)` are [rank](../../glossary.md#term-rank)-specific conveniences for
one-dimensional [pooling](../../glossary.md#term-pooling) in NCW order: [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters), [channel](../../glossary.md#term-channel), width. They do not add maximum- or
average-Pool1d [operation](../../glossary.md#operation) kinds. Each successful call creates exactly this visible composition:

```text
[N, C, W] --EXPAND_DIMS(axis 2)--> [N, C, 1, W]
          --MAX_POOL2D or AVERAGE_POOL2D--> [N, C, 1, W_out]
          --SQUEEZE(axis 2)--> [N, C, W_out]
```

`MaxPool1dAttrs` and `AveragePool1dAttrs` are [immutable](../../glossary.md#term-immutable) public composition [parameters](../../glossary.md#term-parameter). Both [carry](../../glossary.md#term-carry)
positive `kernelWidth`, `strideWidth`, and `dilationWidth`, non-negative symmetric
`paddingWidth`, and literal `ceilMode`. Construction maps those components to a fresh matching
Pool2d attributes value:

```text
height: kernel=1,           stride=1,           padding=0,            dilation=1
width:  kernel=kernelWidth, stride=strideWidth, padding=paddingWidth, dilation=dilationWidth
```

The [singleton](../../glossary.md#term-singleton) height has output [extent](../../glossary.md#term-extent) one in both floor and literal-ceil modes. Width therefore
uses the existing Pool2d formula without alteration:

For integer width $W\geq0$, kernel count $k>0$, dilation $d>0$, padding $p\geq0$
per side and stride $s>0$, define effective span $e$ and start-distance $n$:

$$
e=d(k-1)+1,\qquad n=W+2p-e,\qquad W_{\mathrm{out}}=\lfloor n/s\rfloor+1.
$$

Valid geometry requires $n\geq0$. With $W=9$, $k=3$, $d=1$, $p=1$, $s=2$,
the example below has $e=3$, $n=8$ and five output positions.

When `ceilMode` is true, the quotient uses literal ceiling division. A static negative numerator
fails before any [producer](../../glossary.md#term-producer) is created. Checked `long` [overflow](../../glossary.md#term-overflow) fails with `ArithmeticException`.
An unresolved width retains the exact symbolic floor- or ceiling-division expression and the
obligation that a later concrete [binding](../../glossary.md#term-binding) make the numerator non-negative. The receiver must be a
floating rank-three [Tensor](../../glossary.md#tensor); null attributes, a non-floating type, and another rank fail locally.

Maximum pooling inherits Pool2d [padding](../../glossary.md#padding) exclusion and increasing logical width order. An
in-bounds [NaN](../../glossary.md#term-nan) is dominant at its first logical [occurrence](../../glossary.md#term-occurrence), [positive zero](../../glossary.md#nan-infinity-and-signed-zero) ranks above [negative
zero](../../glossary.md#nan-infinity-and-signed-zero), equal values retain their first occurrence, and an all-padding window produces [negative
infinity](../../glossary.md#nan-infinity-and-signed-zero). Average pooling inherits the fixed count-padding divisor, which reduces from
`1 * kernelWidth` to `kernelWidth`; padding contributes positive zero while still counting.
All four low types and [FLOAT32](../../glossary.md#term-float32) average accumulation and division use FLOAT32; [FLOAT64](../../glossary.md#term-float64) uses FLOAT64,
and a low result narrows once. The existing Pool2d NaN, [infinity](../../glossary.md#term-infinity), [signed-zero](../../glossary.md#nan-infinity-and-signed-zero), [rounding](../../glossary.md#term-rounding),
[reassociation](../../glossary.md#term-reassociation), and all-padding policies otherwise apply unchanged.

<a id="pool1d-shape-and-provenance-example"></a>
#### Pool1d [Shape](../../glossary.md#shape) and [provenance](../../glossary.md#provenance) example

The goal is to inspect maximum and average Pool1d [metadata](../../glossary.md#term-metadata) for input `[2, 3, 9]`. A width-three
[kernel](../../glossary.md#kernel), [stride](../../glossary.md#term-stride) two, one [padding](../../glossary.md#padding) position per side, unit [dilation](../../glossary.md#term-dilation), and floor mode produce width
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
// Inspect the composed rank edits and Pool2d; no pooling arithmetic happens here.
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

Each call returns a fresh [canonical](../../glossary.md#term-canonical-form), unlabeled, storage-free [Tensor](../../glossary.md#tensor) with unresolved [layout](../../glossary.md#layout). It
retains the input [data type](../../glossary.md#data-type--datatype), [gradient](../../glossary.md#term-gradient) request, and exact [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[channel](../../glossary.md#term-channel) [Dimension](../../glossary.md#dimension) references while
recording three fresh one-output [producers](../../glossary.md#term-producer) in order. [Compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings), [descriptor](../../glossary.md#term-descriptor) [validation](../../glossary.md#term-validation), and
gradients consequently see the ordinary [rank](../../glossary.md#term-rank)-edit and Pool2d occurrences. Maximum gradients
inherit Pool2d's reconstructed first eligible winner; average gradients inherit its fixed divisor
and overlap fold. There is no Pool1d [signature](../../glossary.md#term-signature), captured [node](../../glossary.md#node), dedicated gradient rule, or saved
maximum-index output.

[Backend](../../glossary.md#backend) support is the conjunction of support for the exact `EXPAND_DIMS`, matching Pool2d, and
`SQUEEZE` occurrences with their actual descriptors and attributes. A backend may recognize the
whole [topology](../../glossary.md#term-topology) as an optimization only if it preserves that meaning; neither this [Model](../../glossary.md#term-model) [API](../../glossary.md#term-application-programming-interface-api) nor
such recognition creates or advertises a Pool1d [operation](../../glossary.md#operation) capability. The current [CPU](../../glossary.md#term-cpu) backend
recognizes only this exact private single-use topology with resolved [singleton](../../glossary.md#term-singleton)-height affine
layouts. It keeps both rank edits virtual and reuses the Pool2d family projection under the
current [schema](../../glossary.md#term-schema)-70 envelope
bytes that the matching direct Pool2d [occurrence](../../glossary.md#term-occurrence) uses. Near matches retain ordinary [decomposition](../../glossary.md#term-decomposition).
The example itself still proves Model metadata and provenance only: constructing it does not read
values, execute [pooling](../../glossary.md#term-pooling), guarantee that CPU recognition will apply, or promise [fusion](../../glossary.md#term-fusion).

<a id="nchw-maximum-pooling-expressions"></a>
### [NCHW](../../glossary.md#term-nchw) maximum-[pooling](../../glossary.md#term-pooling) expressions

`input.maxPool2d(attrs)` constructs one `MAX_POOL2D` expression over a floating [rank](../../glossary.md#term-rank)-four input in
NCHW [axis](../../glossary.md#term-axis) order: [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) (`N`), [channel](../../glossary.md#term-channel) (`C`), height (`H`), then width (`W`). Pooling moves a
two-dimensional window over each batch/channel plane and requests the greatest in-bounds sampled
value. Current construction records that meaning and derives [metadata](../../glossary.md#term-metadata); it does not read input
values or perform pooling.

The input and result contracts are:

| Value | Required [Shape](../../glossary.md#shape) and type | Result meaning |
|---|---|---|
| receiver input | `[N, C, H, W]`; floating, including [FP8](../../glossary.md#term-fp8) | one rank-four NCHW source |
| result | `[N, C, H_out, W_out]`; exact input type | exact input `N` and `C` [Dimension](../../glossary.md#dimension) references plus derived [spatial](../../glossary.md#term-spatial-axes) Dimensions |

`MaxPool2dAttrs` carries the complete intrinsic window geometry. Each successful call retains the
exact attributes reference in [provenance](../../glossary.md#provenance).

| Attribute | Constraint | Meaning |
|---|---|---|
| `kernelHeight`, `kernelWidth` | positive `long` | number of logical [samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters) in the corresponding [kernel](../../glossary.md#kernel) axis |
| `strideHeight`, `strideWidth` | positive `long` | distance between consecutive window starts |
| `paddingHeight`, `paddingWidth` | non-negative `long` | symmetric [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) positions per side; these positions are excluded from maximum selection |
| `dilationHeight`, `dilationWidth` | positive `long` | spacing between consecutive kernel samples |
| `ceilMode` | [boolean](../../glossary.md#data-type--datatype) | use the literal ceiling grid when true and the floor grid when false |

For either [spatial axis](../../glossary.md#spatial-axes-channels-and-window-sampling), let `D` be its input [extent](../../glossary.md#term-extent), `k` the kernel sample count, `p` the
[padding](../../glossary.md#padding) on each side, `d` the [dilation](../../glossary.md#term-dilation), and `s` the [stride](../../glossary.md#term-stride):

$$
e=d(k-1)+1,\qquad n=D+2p-e,\qquad
Q_{\mathrm{floor}}=\lfloor n/s\rfloor+1,\qquad
Q_{\mathrm{ceil}}=\lceil n/s\rceil+1.
$$

All quantities count integer positions: $D,p\geq0$, $k,d,s>0$, and valid geometry
requires $n\geq0$. Here $e$ is effective span and $Q$ the output count for that axis.
For the example below, $D=2$, $k=d=1$, $p=0$, $s=3$, hence $n=1$:
floor mode gives one position, literal ceiling mode two.

All literal geometry uses checked signed-`long` arithmetic. A static negative numerator fails
before result construction. For an unresolved extent, construction retains the [canonical](../../glossary.md#term-canonical-form)
`addConstant`, floor-or-ceiling divide, then `addConstant` expression. A later concrete [binding](../../glossary.md#term-binding)
must satisfy the retained obligation that the numerator is non-negative.

<a id="static-floor-and-literal-ceil-example"></a>
#### Static floor and literal-ceil example

The goal is to compare floor and ceil grids for input [Shape](../../glossary.md#shape) `[1, 1, 2, 2]` with a one-[sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters)
[kernel](../../glossary.md#kernel), [stride](../../glossary.md#term-stride) three, no [padding](../../glossary.md#padding), and unit [dilation](../../glossary.md#term-dilation).

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

        // Literal ceil retains the terminal grid position even when its window is all padding.
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

For each [axis](../../glossary.md#term-axis), the numerator is `2 - 1 = 1`. Floor division gives zero, then adding one gives
one output position. Ceiling division gives one, then adding one gives two. The second position
starts beyond the input, but the selected literal ceil grid does not apply a terminal-window
decrement. Its all-padding window has the eventual value [negative infinity](../../glossary.md#nan-infinity-and-signed-zero). This example proves
Shape [metadata](../../glossary.md#term-metadata) only; current [model](../../glossary.md#term-model) code does not calculate either output [Tensor](../../glossary.md#tensor)'s values.

<a id="symbolic-floor-and-ceil-example"></a>
#### Symbolic floor and ceil example

For input [Shape](../../glossary.md#shape) `[N, C, H, W]`, [kernel](../../glossary.md#kernel) `3 x 5`, [stride](../../glossary.md#term-stride) `2 x 3`, [padding](../../glossary.md#padding) `1 x 0`, and [dilation](../../glossary.md#term-dilation)
`2 x 1`, the effective [kernels](../../glossary.md#kernel) are five in both [axes](../../glossary.md#term-axis). Construction retains:

```text
floor Shape = [N, C, floor((H - 3) / 2) + 1, floor((W - 5) / 3) + 1]
ceil Shape  = [N, C, ceil((H - 3) / 2) + 1,  ceil((W - 5) / 3) + 1]
```

These are exact symbolic expressions, not estimated extents. A later [binding](../../glossary.md#term-binding) must reject any
concrete `H` below three or `W` below five because that would make a numerator negative.

<a id="extrema-and-padding-policy"></a>
#### [Extrema](../../glossary.md#saturation-pole-and-extremum) and [padding](../../glossary.md#padding) policy

For output [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) `o`, a [spatial](../../glossary.md#term-spatial-axes) window starts at `o * stride - padding`. Logical [kernel](../../glossary.md#kernel)
[sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) `r` reads coordinate `start + r * dilation`. Samples outside the input are excluded rather
than converted to zero. If no sampled coordinate is in bounds, the result is exact [negative
infinity](../../glossary.md#nan-infinity-and-signed-zero) in the input type. Otherwise selection follows these rules:

- any [not-a-number](../../glossary.md#nan-infinity-and-signed-zero) ([NaN](../../glossary.md#term-nan)) candidate wins over every non-NaN, so NaN propagates;
- multiple NaNs select the first logical kernel sample, although NaN payload/sign and signaling
  preservation are unspecified;<span data-glossary-navigation="true"> ([NaN payload glossary](../../glossary.md#nan-infinity-and-signed-zero))</span>
- [positive zero](../../glossary.md#nan-infinity-and-signed-zero) wins over [negative zero](../../glossary.md#nan-infinity-and-signed-zero);
- [infinities](../../glossary.md#nan-infinity-and-signed-zero) use ordinary numerical order; and
- other equal candidates select the first sample in increasing kernel-height, then kernel-width,
  order.

Every success returns one fresh, unlabeled, storage-free [Tensor](../../glossary.md#tensor) with unresolved [layout](../../glossary.md#layout). It
retains the exact input type, `N` and `C` [Dimension](../../glossary.md#dimension) references, and `requiresGrad` request. Its
one-output [provenance](../../glossary.md#provenance) has index zero and ordered inputs `[input]`. The [gradient](../../glossary.md#term-gradient) request is
[Model](../../glossary.md#term-model) [metadata](../../glossary.md#term-metadata), not a Model-owned gradient implementation. Package-private [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings),
[descriptor](../../glossary.md#term-descriptor)-visible proof or retention of deferred constraints, and compiler-owned first-order
input-[cotangent](../../glossary.md#cotangent) construction are current. The formula reconstructs the exact first eligible
logical winner from the original input and same-[occurrence](../../glossary.md#term-occurrence) public output, including padding,
NaN, and [signed-zero](../../glossary.md#nan-infinity-and-signed-zero) policy; it adds no saved-index output. Qualified forward and generated
backward occurrences already lower, [prepare](../../glossary.md#prepare) and execute through [backend](../../glossary.md#backend) [owners](../../glossary.md#term-owner). Construction
itself does not perform these stages or grant arbitrary dynamic [pooling](../../glossary.md#term-pooling) and [decomposition](../../glossary.md#term-decomposition).

<a id="nchw-average-pooling-expressions"></a>
### [NCHW](../../glossary.md#term-nchw) average-[pooling](../../glossary.md#term-pooling) expressions

`input.averagePool2d(attrs)` is the [canonical](../../glossary.md#term-canonical-form) average-pooling receiver expression. It constructs
one `AVERAGE_POOL2D` [occurrence](../../glossary.md#term-occurrence) for a floating [rank](../../glossary.md#term-rank)-four NCHW input. Each output combines one
logical two-dimensional window using a fixed count-[padding](../../glossary.md#padding) divisor: every [kernel](../../glossary.md#kernel) position counts,
including positions outside the input. Current construction records this meaning and derives
[metadata](../../glossary.md#term-metadata); it does not read values or calculate an average.

| Value | Required [Shape](../../glossary.md#shape) and type | Result meaning |
|---|---|---|
| receiver input | `[N, C, H, W]`; floating, including [FP8](../../glossary.md#term-fp8) | one rank-four NCHW source |
| result | `[N, C, H_out, W_out]`; exact input type | exact input `N` and `C` [Dimension](../../glossary.md#dimension) references plus derived [spatial](../../glossary.md#term-spatial-axes) Dimensions |

`AveragePool2dAttrs` is distinct from `MaxPool2dAttrs` because average pooling owns divisor and
accumulation semantics rather than [extrema](../../glossary.md#saturation-pole-and-extremum) semantics. Each successful call retains the exact
attributes reference in [provenance](../../glossary.md#provenance).

| Attribute or policy | Constraint | Meaning |
|---|---|---|
| `kernelHeight`, `kernelWidth` | positive `long` | logical [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) counts; their positive mathematical product is the fixed divisor |
| `strideHeight`, `strideWidth` | positive `long` | distance between consecutive window starts |
| `paddingHeight`, `paddingWidth` | non-negative `long` | symmetric positions per side; each contributes conceptual [positive zero](../../glossary.md#nan-infinity-and-signed-zero) and still counts in the divisor |
| `dilationHeight`, `dilationWidth` | positive `long` | spacing between consecutive kernel positions; it does not change the divisor |
| `ceilMode` | [boolean](../../glossary.md#data-type--datatype) | use the literal ceiling grid when true and the floor grid when false |
| accumulation and division | fixed by input type | All four low types and [FLOAT32](../../glossary.md#term-float32) use FLOAT32; [FLOAT64](../../glossary.md#term-float64) uses FLOAT64; divide the accumulated sum once |

There is no count-padding option, valid-sample averaging mode, or divisor override. For either
[spatial axis](../../glossary.md#spatial-axes-channels-and-window-sampling), input [extent](../../glossary.md#term-extent) `D`, kernel-position count `k`, padding per side `p`, [dilation](../../glossary.md#term-dilation) `d`, and
[stride](../../glossary.md#term-stride) `s` produce:

$$
e=d(k-1)+1,\qquad n=D+2p-e,\qquad
Q_{\mathrm{floor}}=\lfloor n/s\rfloor+1,\qquad
Q_{\mathrm{ceil}}=\lceil n/s\rceil+1.
$$

The integer effective span is $e$ and output count is $Q$; $D,p\geq0$, $k,d,s>0$,
and $n\geq0$ is required. These are the same geometry rules as maximum pooling,
not the average's numeric divisor.

Literal geometry uses checked signed-`long` arithmetic. A static negative numerator fails before
result construction. An unresolved extent retains the canonical add, floor-or-ceiling divide,
then add expression plus the obligation that a later concrete [binding](../../glossary.md#term-binding) makes the numerator
non-negative.

<a id="static-floor-literal-ceil-and-symbolic-examples"></a>
#### Static floor, literal-ceil, and symbolic examples

For input [Shape](../../glossary.md#shape) `[1, 1, 2, 2]`, a `1 x 1` [kernel](../../glossary.md#kernel), `3 x 3` [stride](../../glossary.md#term-stride), zero [padding](../../glossary.md#padding), and unit [dilation](../../glossary.md#term-dilation),
floor mode produces Shape `[1, 1, 1, 1]`, while literal ceil mode produces
`[1, 1, 2, 2]`. The terminal ceil-grid positions begin in trailing padding and therefore describe
all-padding windows; average [pooling](../../glossary.md#term-pooling) retains them and their eventual value is [positive zero](../../glossary.md#nan-infinity-and-signed-zero).

For symbolic input Shape `[N, C, H, W]`, kernel `3 x 5`, stride `2 x 3`, padding `1 x 0`, and
dilation `2 x 1`, both effective [kernels](../../glossary.md#kernel) are five. Construction retains these exact expressions:

```text
floor Shape = [N, C, floor((H - 3) / 2) + 1, floor((W - 5) / 3) + 1]
ceil Shape  = [N, C, ceil((H - 3) / 2) + 1,  ceil((W - 5) / 3) + 1]
```

These examples prove current Shape [metadata](../../glossary.md#term-metadata) only. They do not bind `H` or `W`, evaluate a window,
or establish [compiler](../../glossary.md#term-compiler) or [backend](../../glossary.md#backend) support.

<a id="count-padding-numerical-example"></a>
#### Count-[padding](../../glossary.md#padding) numerical example

Consider one conceptual [FLOAT32](../../glossary.md#term-float32) input plane with values `[[1, 2], [3, 4]]`, a `2 x 2` [kernel](../../glossary.md#kernel),
unit [stride](../../glossary.md#term-stride) and [dilation](../../glossary.md#term-dilation), and padding one. The top-left window contains input value `1` and three
padding positions. The selected average is:

$$
\text{numerator}=1+0+0+0=1,\qquad
\text{divisor}=2\times2=4,\qquad
\text{result}=1/4=0.25.
$$

The four terms are the four kernel positions, not four valid input samples. This exact
worked case illustrates the fixed divisor; other floating sums obey the rounding rules below.

Valid-[sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) averaging would divide by one and return `1.0`; that is deliberately not this
[operation](../../glossary.md#operation). The calculation explains eventual semantics. Current [model](../../glossary.md#term-model) construction does not
produce `0.25` or allocate result storage.

<a id="numerical-empty-and-metadata-policy"></a>
#### Numerical, empty, and [metadata](../../glossary.md#term-metadata) policy

Every in-bounds value contributes once to the numerator and every out-of-bounds position
contributes exact [positive zero](../../glossary.md#nan-infinity-and-signed-zero). All four low types and [FLOAT32](../../glossary.md#term-float32) accumulate and divide in FLOAT32;
[FLOAT64](../../glossary.md#term-float64) uses FLOAT64. A low result narrows once after division. The [accumulator](../../glossary.md#accumulator-type)-domain sum is
divided exactly once by `kernelHeight * kernelWidth`.

- Any in-bounds [not-a-number](../../glossary.md#nan-infinity-and-signed-zero) ([NaN](../../glossary.md#term-nan)) makes the result NaN.
- Positive and negative infinity together make the result NaN; otherwise an infinity retains its
  sign.<span data-glossary-navigation="true"> ([negative infinity glossary](../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../glossary.md#term-infinity))</span>
- An exact-zero finite mean is negative zero only when every divisor contribution is an in-bounds
  negative zero. Cancellation, any positive zero, or any padding contribution produces positive
  zero.<span data-glossary-navigation="true"> ([mean glossary](../../glossary.md#term-mean); [negative zero glossary](../../glossary.md#nan-infinity-and-signed-zero); [Cancellation glossary](../../glossary.md#term-cancellation))</span>
- An all-[padding](../../glossary.md#padding) window therefore returns exact positive zero.
- Empty batch or channel extents are valid and produce no values. Positive kernel counts mean
  there is no zero-divisor case.<span data-glossary-navigation="true"> ([batch glossary](../../glossary.md#term-batch-machine-learning-and-hyperparameters); [channel glossary](../../glossary.md#term-channel))</span>

The [aggregate floor](../../glossary.md#term-aggregate-floor) permits rounded summation trees while retaining every mapped contributor
and the fixed divisor. Arithmetic sites obey their declared working [dtype](../../glossary.md#data-type--datatype), [rounding](../../glossary.md#term-rounding) and [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz)
rules; no [finite](../../glossary.md#term-finite) [tolerance](../../glossary.md#term-tolerance) changes membership, padding or special-value requirements.
[Backend](../../glossary.md#backend)/route finite tolerances are test-only, not a later production policy. Traversal order,
[NaN payload](../../glossary.md#nan-infinity-and-signed-zero)/sign preservation and cross-backend bitwise [identity](../../glossary.md#term-identity-meanings) are not promised.

Every success returns one fresh, unlabeled, storage-free [Tensor](../../glossary.md#tensor) with unresolved [layout](../../glossary.md#layout). It retains
the exact input type, `N` and `C` [Dimension](../../glossary.md#dimension) references, and `requiresGrad` request. Its one-output
[provenance](../../glossary.md#provenance) has index zero and ordered inputs `[input]`. The [gradient](../../glossary.md#term-gradient) request is metadata only.
Package-private [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings) and [descriptor](../../glossary.md#term-descriptor)-visible proof or retention of deferred constraints
are current, as is compiler-owned first-order input-[cotangent](../../glossary.md#cotangent) construction through a logical typed
fixed [kernel](../../glossary.md#kernel) count and public expansion/fold expressions. Qualified occurrences already have
backend [lowering](../../glossary.md#lowering), preparation and execution. Their [derivative](../../glossary.md#term-derivative) [topology](../../glossary.md#term-topology) must qualify in full,
preserving the fixed divisor and special-value policy; construction adds no implicit route
or permission for an arbitrary dynamic [decomposition](../../glossary.md#term-decomposition).

<a id="ncdhw-pool3d-expressions"></a>
### [NCDHW](../../glossary.md#conv3d--ncdhw) [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) expressions

`input.maxPool3d(maxAttrs)` and `input.averagePool3d(averageAttrs)` construct one first-class
three-dimensional [pooling](../../glossary.md#term-pooling) [occurrence](../../glossary.md#term-occurrence) over a floating [rank](../../glossary.md#term-rank)-five [tensor](../../glossary.md#tensor) in NCDHW order: [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)
(`N`), [channel](../../glossary.md#term-channel) (`C`), depth (`D`), height (`H`), then width (`W`). [Model](../../glossary.md#term-model) construction records
geometry, numerical meaning, and [provenance](../../glossary.md#provenance) only. It does not read input values or produce a
materialized numerical result.

| Value | Required [Shape](../../glossary.md#shape) and type | Result meaning |
|---|---|---|
| receiver input | `[N, C, D, H, W]`; floating, including [FP8](../../glossary.md#term-fp8) | one rank-five NCDHW source |
| result | `[N, C, D_out, H_out, W_out]`; exact input type | exact input `N` and `C` [Dimension](../../glossary.md#dimension) references plus three derived [spatial](../../glossary.md#term-spatial-axes) Dimensions |

`MaxPool3dAttrs` and `AveragePool3dAttrs` are separate [immutable](../../glossary.md#term-immutable) thirteen-component records. In
each record, [kernel](../../glossary.md#kernel), [stride](../../glossary.md#term-stride), [padding](../../glossary.md#padding), and [dilation](../../glossary.md#term-dilation) fields use depth, height, then width order;
`ceilMode` follows those twelve values. Kernel, stride, and dilation components are positive.
Symmetric padding per side is non-negative. There is no geometry array, asymmetric intrinsic
padding, adaptive/global mode, valid-[sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) average, or divisor override.

For each spatial input [extent](../../glossary.md#term-extent) `X`, kernel count `k`, symmetric padding per side `p`, dilation `d`,
and stride `s`:

$$
e=d(k-1)+1,\qquad n=X+2p-e,\qquad
Q_{\mathrm{floor}}=\lfloor n/s\rfloor+1,\qquad
Q_{\mathrm{ceil}}=\lceil n/s\rceil+1.
$$

For each depth/height/width axis independently, $e$ is effective span and $Q$ output count.
All sizes are integers with $X,p\geq0$, $k,d,s>0$ and $n\geq0$.
For the depth example below, $X=4$, $k=2$, $p=1$, $d=1$, $s=2$ give
$n=4$ and three output positions.

`ceilMode == true` selects the literal ceiling formula independently for depth, height, and
width. It does not remove a terminal window that begins wholly in trailing padding. Static
arithmetic is checked in signed `long`, and a static negative numerator fails before [producer](../../glossary.md#term-producer)
creation. An unresolved extent retains the [canonical](../../glossary.md#term-canonical-form) add, matching floor-or-ceiling divide, and
final add expression; a later concrete [binding](../../glossary.md#term-binding) must still make its numerator non-negative.
[Validation](../../glossary.md#term-validation) and spatial derivation proceed depth, then height, then width.

Maximum pooling excludes out-of-bounds positions from selection. Eligible logical kernel
[coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) are ordered by increasing depth, then height, then width. An in-bounds [not-a-number](../../glossary.md#nan-infinity-and-signed-zero)
([NaN](../../glossary.md#term-nan)) dominates every non-NaN and the first eligible NaN wins. Otherwise ordinary maximum
applies, [positive zero](../../glossary.md#nan-infinity-and-signed-zero) ranks above [negative zero](../../glossary.md#nan-infinity-and-signed-zero), [infinities](../../glossary.md#nan-infinity-and-signed-zero) use ordinary order, and equal
candidates retain the first eligible occurrence. A window with no in-bounds coordinate has exact
[negative infinity](../../glossary.md#nan-infinity-and-signed-zero) in the input type. A selected non-NaN value keeps its represented input value;
there is no [accumulator](../../glossary.md#accumulator-type) or saved-index output.

Average pooling uses the fixed positive mathematical divisor
`kernelDepth * kernelHeight * kernelWidth`. Every logical kernel position counts. An in-bounds
coordinate contributes its value once; an out-of-bounds coordinate contributes conceptual
positive zero and still counts. Dilation changes coordinates, not the divisor, and Model
construction does not materialize the three-factor product as a `long`. [BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16), and
[FLOAT32](../../glossary.md#term-float32) accumulate and divide in FLOAT32; [FLOAT64](../../glossary.md#term-float64) uses FLOAT64, the sum is divided once, and a low
result narrows once after division. [Finite](../../glossary.md#term-finite) summation may be reassociated.

For average pooling, an in-bounds NaN produces NaN. Opposing [infinity](../../glossary.md#term-infinity) signs produce NaN;
otherwise an infinity retains its sign. An exact-zero finite [mean](../../glossary.md#term-mean) is negative zero only when
every divisor position is an in-bounds negative zero. [Cancellation](../../glossary.md#term-cancellation), any positive zero, or any
padding contribution produces positive zero, so an all-padding window produces positive zero.
No traversal, bitwise cross-[backend](../../glossary.md#backend) result, or NaN representation is promised.

<a id="pool3d-metadata-example"></a>
#### [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) [metadata](../../glossary.md#term-metadata) example

The goal is to inspect metadata for input [Shape](../../glossary.md#shape) `[1, 2, 4, 5, 6]`. A `2 x 3 x 2` [kernel](../../glossary.md#kernel),
[stride](../../glossary.md#term-stride) `(2, 2, 3)`, symmetric [padding](../../glossary.md#padding) `(1, 1, 1)`, and unit [dilation](../../glossary.md#term-dilation) produce output Shape
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

// Both calls share geometry, but maximum exclusion and average divisors remain different.
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

Each call creates a fresh, unlabeled, storage-free [canonical](../../glossary.md#term-canonical-form) [Tensor](../../glossary.md#tensor) with unresolved [layout](../../glossary.md#layout). The
result retains the exact input type, [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[channel](../../glossary.md#term-channel) [Dimension](../../glossary.md#dimension) references, and [gradient](../../glossary.md#term-gradient)-request
metadata. [Provenance](../../glossary.md#provenance) retains the exact supplied attributes reference, ordered input `[input]`,
output index zero, and the canonical `producer.output(0)` wrapper. The example proves those
current [Model](../../glossary.md#term-model) facts only. Complete [Compiler](../../glossary.md#term-compiler) tasks 0006B1 and 0006B2 [capture](../../glossary.md#term-capture-meanings), validate, and
differentiate both Pool3d [signatures](../../glossary.md#term-signature) through ordinary public Tensor expressions. Average Pool3d
uses its fixed logical kernel divisor and overlap-accumulating `fold3d`. Maximum Pool3d uses the
exact same-[occurrence](../../glossary.md#term-occurrence) output and an explicit in-bounds eligibility [mask](../../glossary.md#term-boolean-mask) so dominant [NaN](../../glossary.md#term-nan),
[signed-zero](../../glossary.md#nan-infinity-and-signed-zero) ordering, real [negative infinity](../../glossary.md#nan-infinity-and-signed-zero), first-winner, and all-padding behavior remain
distinguishable; its selected mask is fixed for a later [derivative](../../glossary.md#term-derivative) stage. Current Model owns the
general `unfold3d`/`fold3d` algebra. [CPU](../../glossary.md#term-cpu) [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) now admit matching input/output gradient
flags and actual Pool3d backward closure through these existing shared rules and window families.
Direct forward execution retains fully static Shapes, resolved non-negative layouts, [injective](../../glossary.md#term-injective)
output layouts and checked geometry. CPU low [pooling](../../glossary.md#term-pooling) forward and selected actual gradients use
the same [owners](../../glossary.md#term-owner) with FLOAT32 arithmetic and final [narrowing](../../glossary.md#term-narrowing). Existing
[scalar](../../glossary.md#term-scalar)/caller-parallel forward emission owns complete output cells with zero workspace or
[materialization](../../glossary.md#materialization); no standalone backward kernel or [vector](../../glossary.md#term-vector)/native/[fused](../../glossary.md#specialization-and-fusion) pooling route was added.
Qualified CPU all-valid/[runtime](../../glossary.md#term-runtime)-length [Scan](../../glossary.md#term-cumulative-scan) composes both pools and their actual reverse bodies
in [homogeneous](../../glossary.md#term-homogeneous) FLOAT32/FLOAT64 and both traversal directions, under existing complete-body,
static-window and terminal restrictions. Ordinary public named-B [admission](../../glossary.md#term-admission) is unchanged.
[Metal](../../glossary.md#term-metal) FLOAT32 admits overlapping Pool2d/Pool3d gradient-bearing windows through the same
closure; original-[carrier](../../glossary.md#term-storage-carrier) Java/raw-native checks retain FLOAT64/low non-overlap restrictions.
No low arithmetic expansion, Metal Scan or fallback follows. These are locally runtime-qualified
capabilities, not joined-[release](../../glossary.md#term-release) acceptance; construction itself still performs no [evaluation](../../glossary.md#term-evaluation).


---

[Previous topic](convolution.md) · [Next topic](sort-topk.md)
