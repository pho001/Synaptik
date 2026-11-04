# Softmax, layer, RMS and inference normalization

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

<a id="softmax-expressions"></a>
### Softmax expressions

The two current softmax methods build shape-preserving normalization metadata along exactly one
input axis:

| Public form | Operation kind | Ideal slice meaning |
|---|---|---|
| `softmax(axis)` | `SoftmaxKind.SOFTMAX` | Maximum-shifted probabilities; a rounded small probability may be zero |
| `logSoftmax(axis)` | `SoftmaxKind.LOG_SOFTMAX` | Maximum-shifted log-probabilities, not logarithms of rounded softmax outputs |

A normalization slice fixes every coordinate except the selected axis. For a rank-two tensor and
axis `1`, each row is therefore normalized independently. The input must be floating, including
either FP8 type; integral and BOOL inputs are rejected before axis validation. A positive
or negative caller axis is normalized exactly once against the input Shape.

Every valid result is a fresh unlabeled, storage-free Tensor. Its descriptor retains the exact
input Shape reference, data type, and `requiresGrad` value, but its layout is unresolved even when
the input layout is resolved. Provenance records the exact requested kind, one
`SoftmaxAttrs(normalizedAxis)`, and exact ordered inputs `[input]`. Construction does not inspect
input values or storage, calculate exponentials, logarithms, sums, or probabilities, select a
finite-precision algorithm, define a gradient rule, capture a graph, decompose the operation, or
execute backend work.

For one ideal slice `[1, 2, 3]`, the requested mathematical values are:

| Method | Approximate result | Relationship |
|---|---|---|
| `softmax(axis)` | `[0.09003057, 0.24472847, 0.66524096]` | The probabilities sum to approximately one. |
| `logSoftmax(axis)` | `[-2.40760596, -1.40760596, -0.40760596]` | Exponentiating each value recovers the corresponding softmax probability. |

These numbers explain the expression's meaning; the current API constructs only metadata for the
request.

For an all-finite slice `[1000, 1000]`, the requested SOFTMAX values are `[0.5, 0.5]` and
LOG_SOFTMAX values are approximately `[-0.69314718, -0.69314718]`: the implementation does not
have to overflow unshifted `exp(1000)`. With `[0, -1000]`, a rounded SOFTMAX output can be
`[1, 0]` while LOG_SOFTMAX remains `[0, -1000]`. Input NaN/infinity and empty-slice rules are
separate; the [Model numerical contract](../../architecture/contracts/foundational-modules.md#composite-inheritance)
defines them, and a backend may admit a narrower runtime value domain.

<a id="complete-expression-construction-example"></a>
#### Complete expression-construction example

<a id="goal-and-inputs-23"></a>
##### Goal and inputs

Construct softmax and log-softmax expressions for a storage-free `FLOAT32` tensor with Shape
`[2, 3]`. Axis `-1` and axis `1` both select the final dimension, so each logical row is one
normalization slice.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class SoftmaxExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));

        Tensor probabilities = input.softmax(-1);
        Tensor logProbabilities = input.logSoftmax(1);
        SoftmaxAttrs probabilitiesAttrs = (SoftmaxAttrs) probabilities
                .provenance().orElseThrow().operation().attrs();

        System.out.println(probabilities.descriptor().shape() == shape);
        System.out.println(probabilities.descriptor().dataType());
        System.out.println(probabilities.descriptor().requiresGrad());
        System.out.println(probabilities.descriptor().layout().isEmpty());
        System.out.println(probabilities.provenance().orElseThrow().operation().kind());
        System.out.println(logProbabilities.provenance().orElseThrow().operation().kind());
        System.out.println(probabilitiesAttrs.axis());
        System.out.println(probabilities.provenance().orElseThrow().inputs().getFirst() == input);
    }
}
```

<a id="meaningful-lines-and-intermediate-results-8"></a>
##### Meaningful lines and intermediate results

- `input.softmax(-1)` normalizes `-1` to axis `1` and records `SOFTMAX`.
- `input.logSoftmax(1)` stores the same normalized axis but records the distinct
  `LOG_SOFTMAX` kind.
- Both descriptors retain the exact `shape`, `FLOAT32`, and the true gradient-eligibility
  metadata while leaving layout unresolved.
- Each provenance value contains exactly `input`; neither expression receives label or storage.

<a id="result-and-interpretation-23"></a>
##### Result and interpretation

The program prints:

```text
true
FLOAT32
true
true
SOFTMAX
LOG_SOFTMAX
1
true
```

The result proves axis normalization, kind selection, descriptor retention, unresolved layout,
and one-input provenance. It does not prove the ideal numerical values above, gradient support,
compiler capture or decomposition, backend support, or execution.

<a id="failures-and-useful-variations-20"></a>
##### Failures and useful variations

- `INT32`, `INT64`, and `BOOL` input fails with `IllegalArgumentException`; no conversion or
  promotion is inserted. This type check occurs before axis validation.
- An axis outside `[-rank, rank - 1]` fails with `IndexOutOfBoundsException`. Every axis is invalid
  for a scalar input.
- Dynamic and zero extents are accepted structurally when the selected axis exists. Repeated
  equivalent requests still return fresh expression identities.

<a id="layer-normalization-expressions"></a>
### Layer-normalization expressions

Layer normalization standardizes each trailing logical slice selected by an exact non-empty
normalized Shape. It is distinct from softmax: layer normalization centers values and divides by
population standard deviation, while softmax converts one axis to probabilities. The two current
receiver signatures are exactly:

```java
public Tensor layerNorm(Shape normalizedShape, ScalarValue epsilon)
public Tensor layerNorm(
        Shape normalizedShape, Tensor scale, Tensor bias, ScalarValue epsilon)
```

For input Shape `[D0, ..., D(R-1)]` and normalized Shape `[N0, ..., N(K-1)]`, normalized axis `j`
corresponds to input axis `R - K + j`. This table shows the two supported forms.

| Form | Ordered producer inputs | Shape requirements | Result type |
|---|---|---|---|
| No affine | `[input]` | `normalizedShape` exactly describes trailing input axes | Exact input type |
| Affine | `[input, scale, bias]` | Same trailing rule; scale and bias Shapes each equal `normalizedShape` exactly | Floating promotion in input, scale, bias order |

Every operand must be floating, including either FP8 type. Epsilon must be finite and strictly
positive. It has the exact result type except that FLOAT16 and FP8 results use exact FLOAT32 epsilon
metadata for its FLOAT32 arithmetic domain. There is no default epsilon, `double` convenience,
axis-array overload, scale-only form, affine broadcasting, or implicit scale or bias Tensor.

For a non-empty slice of `N` values, the semantic formula is:

```text
mean           = sum_i(x_i) / N
variance       = sum_i((x_i - mean) * (x_i - mean)) / N
standardized_i = (x_i - mean) / sqrt(variance + epsilon)
output_i       = standardized_i                         // no affine
output_i       = standardized_i * scale_i + bias_i      // affine
```

This is population variance, so correction is zero. Epsilon is inside the square root. Scale and
bias coordinates span the normalized Shape and are reused for every leading slice.

<a id="finite-numerical-example"></a>
#### Finite numerical example

The goal is to illustrate the selected formula, not current execution. For one FLOAT64 slice
`[1, 2, 3]`, normalized Shape `[3]`, and epsilon `0.00001`, mean is `2`, population variance is
`2 / 3`, and the denominator is `sqrt(2 / 3 + 0.00001)`.

```text
no-affine result  ~= [-1.2247356859, 0.0, 1.2247356859]
scale = 2.0
bias  = 0.5
affine result     ~= [-1.9494713718, 0.5, 2.9494713718]
```

This establishes centering, population correction, epsilon placement, and affine order. It does
not promise bitwise rounding or a reduction traversal. Conforming execution may reassociate finite
arithmetic under the Model contract. All four low types and FLOAT32
results accumulate mean and variance in FLOAT32; FLOAT64 results use FLOAT64. Affine multiply and
add occur in the result arithmetic domain with one final low-type narrowing.

<a id="static-checks-and-symbolic-deferral"></a>
#### Static checks and symbolic deferral

For input Shape `[2, 3, 4]`, normalized Shape `[3, 4]` is valid. Normalized Shape `[5, 4]` fails
locally because its first extent conflicts with the known trailing input extent:

```text
layerNorm normalized dimension mismatch at normalized axis 0: input=StaticDimension[size=3], normalized=StaticDimension[size=5]
```

Structurally equal symbolic dimensions pass immediately. If a trailing input dimension is `N`
and its normalized dimension is a different unresolved symbol `M`, construction also succeeds
because the exact result Shape is still the input Shape. Compiler verification or concrete
binding must prove `N == M`; model construction does not invent a replacement extent. Scale and
bias are stricter: each complete Shape must structurally equal the normalized Shape, so ordinary
broadcasting is insufficient.

<a id="empty-and-special-value-policy"></a>
#### Empty and special-value policy

A static zero normalized extent is valid when the corresponding input extent is zero. A zero
leading extent is also valid. The result is then empty, contains no normalized values, and exposes
no mean, variance, NaN, infinity, or signed-zero result.

Within a non-empty slice, any NaN or either infinity makes every standardized value in that slice
NaN. A finite constant slice, including mixed positive and negative zero, has positive-zero
variance and exact positive-zero standardized values. The affine form then applies ordinary
floating multiply/add class behavior: infinite scale times standardized zero is NaN, while a
finite product plus one infinite bias retains that bias infinity. Overflow, underflow, and finite
non-zero rounding follow the selected accumulator and result formats; NaN payloads and
cross-backend bitwise identity are not promised.

<a id="result-identity-and-lifecycle-boundary"></a>
#### Result identity and lifecycle boundary

Every valid call creates one fresh, unlabeled, storage-free, layout-unresolved result and retains
the exact input Shape reference. Gradient eligibility is the input flag for no-affine construction
and the logical OR of input, scale, and bias flags for affine construction. One producer has one
output at index zero and exact inputs `[input]` or `[input, scale, bias]`; no saved mean, variance,
inverse standard deviation, sibling result, hidden affine constant, or extra output is created.

These facts describe what the Model constructor does, not the availability of downstream stages.
Compiler capture and revalidation are implemented, as are selected first-order input/scale/bias
cotangents through [NormalizationGradientRules](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/NormalizationGradientRules.java).
That owner constructs ordinary expressions; LayerNorm has no hidden saved outputs to retrieve.
Qualified occurrences lower, prepare and execute through the current backend owners.
Calling `layerNorm` itself performs none of those steps and does not read stored values.

<a id="rms-normalization-expressions"></a>
### RMS-normalization expressions

Root-mean-square (RMS) normalization divides each value in an exact non-empty trailing slice by
the square root of that slice's uncentered mean square plus epsilon. Unlike layer normalization,
it does not subtract a mean, calculate variance, or accept bias. The two current receiver methods
are exactly:

```java
public Tensor rmsNorm(Shape normalizedShape, ScalarValue epsilon)
public Tensor rmsNorm(Shape normalizedShape, Tensor scale, ScalarValue epsilon)
```

The operands and Shape relationships are:

| Form | Ordered inputs | Required Shapes | Result type |
|---|---|---|---|
| No scale | `[input]` | `normalizedShape` exactly describes a non-empty trailing input Shape | Exact input type |
| Scaled | `[input, scale]` | Same trailing rule; scale Shape exactly equals `normalizedShape` and is not broadcast | Ordered floating promotion of input then scale |

Both forms accept every floating type, including FP8. Epsilon must be finite and
strictly positive. It has the exact result type except that FLOAT16 and FP8 results use FLOAT32
epsilon metadata. There is no default epsilon. All four low types and FLOAT32 results accumulate
squares and sums in FLOAT32; FLOAT64 results accumulate them in FLOAT64.

For a non-empty slice of `N` values, the mathematical contract is:

```text
meanSquare   = sum_i(x_i * x_i) / N
rms          = sqrt(meanSquare + epsilon)
normalized_i = x_i / rms
output_i     = normalized_i                 // no-scale form
output_i     = normalized_i * scale_i       // scaled form
```

For FLOAT64 input `[1, 2, 3]`, normalized Shape `[3]`, and epsilon `1e-5`, `meanSquare` is
`14 / 3`, `rms` is approximately `2.1602492140182963`, and the no-scale result is approximately
`[0.4629095539120194, 0.9258191078240388, 1.3887286617360581]`. A scale value `2` at the final
coordinate makes that coordinate approximately `2.7774573234721163`. This example states the
mathematical contract; constructing the Tensor does not calculate these values.

The normalized Shape must have positive rank no greater than input rank. Known static trailing
extents must match. For input Shape `[2, 3, 4]`, normalized Shape `[5, 4]` fails during local
construction:

```text
rmsNorm normalized dimension mismatch at normalized axis 0: input=StaticDimension[size=3], normalized=StaticDimension[size=5]
```

When corresponding unequal Dimensions are unresolved, local construction defers their equality
proof because the result Shape remains the exact input Shape. Current package-private compiler
verification represents the equality and proves, rejects, or retains it without binding either
Dimension. Scale is stricter at construction: its complete Shape must be structurally equal to
`normalizedShape`, including dynamic or expression structure.

A corresponding zero normalized extent or any zero leading extent makes the entire result empty;
no divisor or root mean square is evaluated. In a non-empty slice, any NaN makes all normalized
values NaN. Without NaN, an infinite input makes the root mean square positive infinity: finite
inputs then produce same-signed zero and infinite inputs produce NaN. Positive epsilon preserves
the sign of finite positive or negative zero before scale. Finite square or sum overflow may also
produce an infinite denominator and same-signed zero for finite numerators. In the scaled form,
ordinary floating multiplication then applies, including NaN scale and infinity-times-zero
behavior. Arithmetic follows the [profile-free primitive-site contract](../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics),
not a blanket equal-or-wider-intermediate permission. Low arithmetic uses FLOAT32 working and
accumulator values with one final ties-to-even narrowing at each declared low output; exact
guards, public/saved boundaries, and special-value classes remain intact. Finite tolerances are
test-only qualification, not production permissions or a whole-output error envelope.

Every successful call produces one fresh, unlabeled, storage-free Tensor with unresolved layout,
the exact input Shape reference, combined operand gradient eligibility, one producer, output index
zero, and ordered provenance `[input]` or `[input, scale]`. Construction allocates exactly one
result identity and does not mutate inputs or create a hidden scale, bias, parameter, saved
statistic, sibling output, or gradient rule.

The call constructs metadata only. Compiler capture, revalidation, deferred-constraint proof
and selected first-order input/scale cotangents are implemented; the shared normalization
formula owner constructs ordinary Tensor expressions. Qualified occurrences have backend
lowering, preparation and runtime execution. [Compiler boundaries](../compiler/index.md)
and the [backend matrix](../../backend-guide/production-support.md) describe those separate gates.

<a id="batch-normalization-inference-expressions"></a>
### Batch-normalization inference expressions

Batch-normalization inference applies explicit estimated per-channel statistics without
calculating or updating them. The sole receiver form is:

```java
public Tensor batchNormInference(
        int channelAxis,
        Tensor scale,
        Tensor bias,
        Tensor runningMean,
        Tensor runningVariance,
        ScalarValue epsilon)
```

This is an explicit stateless five-input occurrence. Its ordered producer roles and Shape rules
are:

| Position | Role | Required Shape |
|---|---|---|
| 0 | receiver input | Rank at least two; exact result Shape |
| 1 | scale | Exact rank-one `[C]` vector |
| 2 | bias | Exact rank-one `[C]` vector |
| 3 | running mean | Exact rank-one `[C]` vector |
| 4 | running variance | Exact rank-one `[C]` vector |

`C` is the input extent at `channelAxis`. The caller axis may be positive or negative and is
normalized against the input rank. It may identify any logical input axis: neither a resolved
layout nor an NCHW or NHWC convention changes its meaning. Each coordinate that shares channel
coordinate `c` uses the same four vector entries.

All five Tensors must be floating, including either FP8 type. Their result type is derived by
floating promotion in exact producer order: input with scale, then bias, running mean, and running
variance. Epsilon must be finite and strictly positive. It has the exact result type except that a
FLOAT16 or FP8 result uses exact FLOAT32 epsilon metadata. No cast producer or hidden default is inserted.

For each output coordinate with channel `c`, the mathematical contract is:

```text
centered     = input - runningMean[c]
denominator  = sqrt(runningVariance[c] + epsilon)
standardized = centered / denominator
output       = standardized * scale[c] + bias[c]
```

Epsilon is inside the square root. The running variance is used directly as the estimated
variance supplied by the caller; inference applies no sample/population correction conversion,
recomputation, clamp, momentum, or mutation. All four low types and FLOAT32 results perform formula
arithmetic in FLOAT32 and round the final low value once to the result format. FLOAT64 results use
FLOAT64. There is no reduction accumulator or configurable computation format.

<a id="finite-numerical-example-1"></a>
#### Finite numerical example

The goal is to show the formula and affine order. For FLOAT64 input `[1, 2]` in one channel,
scale `2`, bias `0.5`, running mean `1`, running variance `4`, and epsilon `1e-5`:

```text
input 1: ((1 - 1) / sqrt(4 + 1e-5)) * 2 + 0.5 = 0.5
input 2: ((2 - 1) / sqrt(4 + 1e-5)) * 2 + 0.5 ~= 1.4999987500023437
```

This establishes direct variance use, epsilon placement, and scale-then-bias order. It does not
show current numerical execution: constructing the expression reads no Tensor values.

<a id="static-checks-symbolic-deferral-and-empty-results"></a>
#### Static checks, symbolic deferral, and empty results

Input rank below two and an out-of-range channel axis fail locally. Each of the four other inputs
must have rank exactly one. For input Shape `[2, 3, 4]` and channel axis `1`, each vector must
describe `[3]`; a static `[4]` scale, for example, fails with:

```text
batchNormInference scale channel dimension mismatch: input=StaticDimension[size=3], scale=StaticDimension[size=4]
```

Structurally equal static, dynamic, or symbolic-expression Dimensions pass. Unequal static
Dimensions fail. If either unequal Dimension is unresolved, construction defers the equality
obligation because the result Shape remains exactly the input Shape. Current package-private
compiler capture and verification represent and prove, reject, or retain that obligation without
binding it; concrete binding remains required before execution.

A static channel extent zero is valid with four `[0]` vectors. A zero on another input axis is
also valid while the vectors continue to describe `C`. Either case produces an empty result, so
no subtraction, square root, division, or affine step is evaluated.

<a id="negative-variance-and-special-values"></a>
#### Negative variance and special values

Construction does not read running-variance values and therefore does not reject a Tensor that
may later contain a negative value. Ordinary floating formula behavior applies:

- a negative radicand produces NaN;
- a zero radicand produces a zero denominator, so zero divided by zero is NaN and a non-zero
  numerator divided by zero is signed infinity;
- positive-infinite variance produces an infinite denominator, so a finite numerator produces
  signed zero and an infinite numerator produces NaN; and
- negative-infinite variance produces NaN.

NaN in any participating value propagates through the coordinate. Infinite subtraction,
division, multiplication, and addition follow ordinary floating value classes, including
infinity divided by infinity and infinity multiplied by zero producing NaN. Arithmetic uses
the declared primitive sites and their family-specific DAZ/FTZ, rounding and fusion permissions;
FLOAT64 forbids arithmetic-site DAZ/FTZ. Low arithmetic uses FLOAT32 working/accumulator values
and final declared-output narrowing, without crossing public or saved boundaries. Finite
test tolerances do not authorize wider intermediates, arbitrary FMA, or whole-result errors.
NaN payload/sign preservation and bitwise cross-backend equality are not promised.

<a id="metadata-provenance-and-lifecycle-boundary"></a>
#### Metadata, provenance, and lifecycle boundary

Every success creates a fresh unlabeled, storage-free Tensor with the exact input Shape reference,
ordered-promoted result type, unresolved layout, and gradient eligibility equal to the logical OR
of all five inputs. One identity-distinct producer has exact ordered inputs
`[input, scale, bias, runningMean, runningVariance]`, one descriptor, and provenance output index
zero. Repeated equal requests remain distinct. No input, statistic, parameter, descriptor, label,
storage, or provenance is mutated.

These facts describe the inference constructor: it has no training/evaluation flag, current-batch
statistic calculation, running-statistic transition, momentum, saved statistic, auxiliary output,
or hidden state. Compiler capture, operand revalidation and deferred-constraint proof are
implemented. Its normalization derivative owner can select contributions for all five floating
inputs and casts completed Shape-correct contributions back to the selected input types.
Qualified occurrences lower, prepare and execute through their backend owners. Construction
itself does not invoke those stages. [Compiler boundaries](../compiler/index.md) explain the handoff; the
formula corresponds to official [ONNX
BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) while retaining
Synaptik's explicit channel axis and exact typed contracts.


---

[Previous topic](fixed-scans.md) · [Next topic](general-scan.md)
