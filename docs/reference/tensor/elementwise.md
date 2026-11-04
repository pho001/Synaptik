# [Unary](../../glossary.md#term-unary) transforms, [scalar](../../glossary.md#term-scalar) arithmetic and clamps

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**[Reader model](../../glossary.md#term-reader-model).** **Unary** means one input; an **[elementwise](../../glossary.md#term-elementwise)** function transforms each
position without mixing it with neighboring positions. A **scalar** [parameter](../../glossary.md#term-parameter) supplies one
same-typed value for all positions. A **clamp** limits values to a range.
An **[activation](../../glossary.md#term-activation)** is a numeric transform used between learned computations: [ReLU](../../glossary.md#term-relu) keeps positive
values and maps negative values to zero, while [sigmoid](../../glossary.md#term-sigmoid) smoothly maps real values toward the
interval from zero to one. A floating **classification** instead returns a true/false [tensor](../../glossary.md#tensor)
identifying [finite](../../glossary.md#term-finite) values, NaNs, or [infinities](../../glossary.md#nan-infinity-and-signed-zero); it is not a machine-learning class [prediction](../../glossary.md#term-prediction).
The exact functions and exceptional-value rules remain in the tables below.

For the function names below, a **reciprocal** is `1/x`; **[exp](../../glossary.md#term-expx)** is the [natural exponential](../../glossary.md#norms-elementary-functions-and-probability-distributions)
with base `e`, approximately `2.71828`; and the natural **logarithm** is its inverse.
Thus `exp(0) = 1` and `log(1) = 0` in exact mathematics. **[Tanh](../../glossary.md#term-tanh)** smoothly maps real inputs
between negative one and one. The [Gaussian error function](../../glossary.md#norms-elementary-functions-and-probability-distributions) **[erf](../../glossary.md#term-erf)** is related to the probability
of values falling within a centered [normal distribution](../../glossary.md#norms-elementary-functions-and-probability-distributions); [GELU](../../glossary.md#gaussian-error-linear-unit--gelu)'s [`Phi`](../../glossary.md#term-phi) is the cumulative
probability up to its [argument](../../glossary.md#term-argument) for a [mean](../../glossary.md#term-mean)-zero, unit-standard-deviation normal distribution.
These explanations orient the reader; the formulas and special-value rules below remain exact.

<a id="unary-numeric-transforms-and-floating-classifications"></a>
### [Unary](../../glossary.md#term-unary) numeric transforms and floating classifications

Nineteen zero-[argument](../../glossary.md#term-argument) unary methods create one-input, floating-preserving [elementwise](../../glossary.md#term-elementwise) semantics:

| Method | Elementwise meaning |
|---|---|
| `abs` | Absolute magnitude. |
| `neg` | Additive inverse. |
| `reciprocal` | Multiplicative reciprocal. |
| [`log`](../../glossary.md#term-log) | [Natural logarithm](../../glossary.md#norms-elementary-functions-and-probability-distributions). |
| `log1p` | Natural logarithm of one plus the input. |
| [`exp`](../../glossary.md#term-expx) | Portable [natural exponential](../../glossary.md#norms-elementary-functions-and-probability-distributions) request. |
| `expm1` | Natural exponential of the input minus one. |
| [`erf`](../../glossary.md#term-erf) | [Gaussian error function](../../glossary.md#norms-elementary-functions-and-probability-distributions). |
| `sqrt` | Principal [square root](../../glossary.md#norms-elementary-functions-and-probability-distributions). |
| `rsqrt` | Reciprocal of the principal square root. |
| `floor` | Greatest integer-valued result not greater than the input. |
| `ceil` | Least integer-valued result not less than the input. |
| `sign` | Numeric negative, zero, or positive classification. |
| `relu` | [Rectified linear unit](../../glossary.md#bias-and-activation). |
| [`sigmoid`](../../glossary.md#term-sigmoid) | Logistic sigmoid. |
| [`tanh`](../../glossary.md#term-tanh) | Portable [hyperbolic tangent](../../glossary.md#bias-and-activation) request. |
| `gelu` | Exact [Gaussian error linear unit](../../glossary.md#gaussian-error-linear-unit--gelu), `x * Phi(x)`. |
| `geluTanhApproximation` | Fixed conventional tanh approximation to [GELU](../../glossary.md#gaussian-error-linear-unit--gelu). |
| `silu` | [Sigmoid linear unit](../../glossary.md#sigmoid-linear-unit--silu), `x * sigmoid(x)`. |

Each method accepts every floating type, including both [FP8](../../glossary.md#term-fp8) types. The result retains the exact
input [data type](../../glossary.md#data-type--datatype), [immutable](../../glossary.md#term-immutable) [`Shape`](../../glossary.md#shape) reference, and `requiresGrad` flag, while leaving [layout](../../glossary.md#layout) unresolved.
That flag is eligibility [metadata](../../glossary.md#term-metadata); it does not assert that a [derivative](../../glossary.md#term-derivative) or backward rule exists.

`rsqrt`, `log1p`, `expm1`, both GELU variants, and [SiLU](../../glossary.md#sigmoid-linear-unit--silu) are first-class transforms, not stored
compositions. [Model](../../glossary.md#term-model) owns one family/[dtype](../../glossary.md#data-type--datatype) result contract for these and the other unary kinds.
The six exact represented-value kinds `ABS`, `NEG`, `SIGN`, `FLOOR`, `CEIL`, and `RELU` have no
arithmetic [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz) permission. Other named [FLOAT32](../../glossary.md#term-float32) and low arithmetic primitive inputs
may use [denormals-are-zero](../../glossary.md#subnormal-daz-ftz-overflow-and-underflow) (DAZ), and results may use [flush-to-zero](../../glossary.md#subnormal-daz-ftz-overflow-and-underflow) (FTZ); [FLOAT64](../../glossary.md#term-float64) forbids both.
Each kind retains its mathematical function, domain, special class, range, and required zero-sign
rules. [Finite](../../glossary.md#term-finite) thresholds are route-qualified in tests, not public ordered-distance or forward-
error envelopes. Composite `RSQRT`, sigmoid, GELU, tanh GELU, and SiLU inherit only their named
primitive sites and exact guards; no final-output [tolerance](../../glossary.md#term-tolerance) applies.

Exact GELU uses `0.5 * x * (1 + erf(x / sqrt(2)))`. Its [constants](../../glossary.md#term-constant), square root, division, `ERF`,
addition, and two multiplications are its complete sites. The explicitly named tanh approximation
uses `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`; its constants, `x*x`, `x^2*x`,
root, division, additions, remaining multiplications, and `TANH` are complete. Sigmoid uses
`1/(1+exp(-x))` for nonnegative input and `exp(x)/(1+exp(x))` for negative input. SiLU uses
`x/(1+exp(-x))` and `x*exp(x)/(1+exp(x))` on the corresponding branches. Comparison, negation,
exponential, additions, multiplication, and divisions are explicit sites. The [API](../../glossary.md#term-application-programming-interface-api) uses [canonical](../../glossary.md#term-canonical-form)
`silu` naming and provides no `swish` [alias](../../glossary.md#aliasing-and-arena). None of these first-class operations gains a
whole-result tolerance envelope.

Their Model-owned special-value meanings are:

| Method | Special-value behavior |
|---|---|
| `rsqrt` | Positive/[negative zero](../../glossary.md#nan-infinity-and-signed-zero) becomes same-signed [infinity](../../glossary.md#term-infinity); [positive infinity](../../glossary.md#nan-infinity-and-signed-zero) becomes [positive zero](../../glossary.md#nan-infinity-and-signed-zero); negative finite values and [negative infinity](../../glossary.md#nan-infinity-and-signed-zero) produce [NaN](../../glossary.md#term-nan); NaN produces NaN. |
| `log1p` | [Signed zero](../../glossary.md#term-signed-zero) is preserved; `-1` produces negative infinity; values below `-1`, including negative infinity, produce NaN; positive infinity remains positive infinity; NaN produces NaN. |
| `expm1` | Signed zero is preserved; negative infinity becomes exactly `-1`; positive infinity remains positive infinity; NaN produces NaN. |
| `gelu` | Negative infinity becomes negative zero by continuous extension; signed zero is preserved; positive infinity remains positive infinity; NaN produces NaN. |
| `geluTanhApproximation` | Negative infinity becomes negative zero by continuous extension; signed zero is preserved; positive infinity remains positive infinity; NaN produces NaN. |
| `silu` | Negative infinity becomes negative zero by continuous extension; signed zero is preserved; positive infinity remains positive infinity; NaN produces NaN. |

These meanings specify results, not [Tensor](../../glossary.md#tensor) construction-time [evaluation](../../glossary.md#term-evaluation) or a [backend capability](../../glossary.md#backend-capability).
[Compiler](../../glossary.md#term-compiler) and [backend](../../glossary.md#backend) work may preserve useful near-zero accuracy for `log1p` and `expm1` because
the model records the transforms without [decomposition](../../glossary.md#term-decomposition).

Three separate floating-classification methods describe [BOOL](../../glossary.md#data-type--datatype) values:

| Method | True exactly when |
|---|---|
| `isFinite` | The input is finite: normal, [subnormal](../../glossary.md#term-subnormal), positive zero, or negative zero. |
| `isNaN` | The input is NaN, independent of sign, quiet/signaling encoding, or [payload](../../glossary.md#artifact-envelope-and-atomic-publication). |
| `isInf` | The input is positive or negative infinity. |

Classification accepts the same four floating input types but always produces `BOOL`, retains the
exact input `Shape`, leaves layout unresolved, and sets `requiresGrad=false`. These methods
are non-[differentiable](../../glossary.md#derivative-and-chain-rule) value classifications, not numeric unary transforms, [trace](../../glossary.md#trace) diagnostics,
[eager](../../glossary.md#term-eager-evaluation) Java booleans, or [validation](../../glossary.md#term-validation) checks. For every represented floating value, exactly one of
the three classifications is true when the [operation](../../glossary.md#operation) is eventually evaluated.

Every valid transform or classification call returns a fresh Tensor with no label or [host
storage](../../glossary.md#host-storage). Its [provenance](../../glossary.md#provenance) contains one parameterless `Operation`, exactly the receiver reference,
one [producer](../../glossary.md#term-producer) output, and output index zero. Model construction reads no values and calculates no
result.

<a id="complete-transform-and-classification-example"></a>
#### Complete transform-and-classification example

<a id="goal-and-inputs-18"></a>
##### Goal and inputs

Build one exact [`GELU`](../../glossary.md#gaussian-error-linear-unit--gelu) transform and one `IS_NAN` classification from a storage-free [`FLOAT32`](../../glossary.md#term-float32)
[Tensor](../../glossary.md#tensor) of [Shape](../../glossary.md#shape) `[2, 3]`. The example observes [metadata](../../glossary.md#term-metadata) only; it does not claim transformed or
classified [element](../../glossary.md#term-element) values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class FloatingMetadataExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));

        Tensor transformed = input.gelu();
        Tensor classified = input.isNaN();
        TensorProvenance transformOrigin = transformed.provenance().orElseThrow();
        TensorProvenance classificationOrigin = classified.provenance().orElseThrow();

        System.out.println("transformType=" + transformed.descriptor().dataType());
        System.out.println("transformGrad=" + transformed.descriptor().requiresGrad());
        System.out.println("transformKind=" + transformOrigin.operation().kind());
        System.out.println("classificationType=" + classified.descriptor().dataType());
        System.out.println("classificationGrad=" + classified.descriptor().requiresGrad());
        System.out.println("classificationKind=" + classificationOrigin.operation().kind());
        System.out.println("sameShape="
                + (transformed.descriptor().shape() == shape
                && classified.descriptor().shape() == shape));
        System.out.println("parameterless="
                + (transformOrigin.operation().attrs() == NoOperationAttrs.INSTANCE
                && classificationOrigin.operation().attrs() == NoOperationAttrs.INSTANCE));
        System.out.println("exactInputs="
                + (transformOrigin.inputs().getFirst() == input
                && classificationOrigin.inputs().getFirst() == input));
        System.out.println("oneInputEach="
                + (transformOrigin.inputs().size() == 1
                && classificationOrigin.inputs().size() == 1));
        System.out.println("outputIndexes="
                + transformOrigin.outputIndex() + "," + classificationOrigin.outputIndex());
    }
}
```

<a id="meaningful-lines-and-intermediate-results-4"></a>
##### Meaningful lines and intermediate results

- `input.gelu()` records one `UnaryElementwiseKind.GELU` occurrence and preserves the input's
  floating type and gradient-eligibility request.<span data-glossary-navigation="true"> ([occurrence glossary](../../glossary.md#term-occurrence); [gradient glossary](../../glossary.md#term-gradient))</span>
- `input.isNaN()` records one distinct `FloatingClassificationKind.IS_NAN` occurrence and fixes
  the result descriptor to non-gradient <span data-glossary-navigation="true"> ([descriptor glossary](../../glossary.md#term-descriptor))</span>[`BOOL`](../../glossary.md#data-type--datatype).
- Both results retain the exact [`shape`](../../glossary.md#shape) and input references, use canonical no-attributes state,
  and select output index zero from a one-output producer.<span data-glossary-navigation="true"> ([canonical glossary](../../glossary.md#term-canonical-form); [producer glossary](../../glossary.md#term-producer))</span>

<a id="result-and-interpretation-18"></a>
##### Result and interpretation

The program prints:

```text
transformType=FLOAT32
transformGrad=true
transformKind=GELU
classificationType=BOOL
classificationGrad=false
classificationKind=IS_NAN
sameShape=true
parameterless=true
exactInputs=true
oneInputEach=true
outputIndexes=0,0
```

The output proves the result-type distinction, exact [operation](../../glossary.md#operation) kinds, [Shape](../../glossary.md#shape) and input retention,
parameterless construction, and one-input, output-index-zero [provenance](../../glossary.md#provenance). It does not prove that
[GELU](../../glossary.md#gaussian-error-linear-unit--gelu) or [NaN](../../glossary.md#term-nan) classification ran, that numerical accuracy or a [gradient](../../glossary.md#term-gradient) rule exists, that a [graph](../../glossary.md#graph)
was captured, or that any [backend](../../glossary.md#backend) supports either operation.

<a id="failures-and-useful-variations-16"></a>
##### Failures and useful variations

- Calling any of these methods on an [`INT32`](../../glossary.md#data-type--datatype), [`INT64`](../../glossary.md#data-type--datatype), or [`BOOL`](../../glossary.md#data-type--datatype) [Tensor](../../glossary.md#tensor) fails with
  `IllegalArgumentException` before result [identity](../../glossary.md#term-identity-meanings) allocation; no implicit cast is inserted.
- Scalar, zero-sized, ordinary static, and dynamic Shapes are accepted. A resolved input layout is
  deliberately not copied to either result.<span data-glossary-navigation="true"> ([Scalar glossary](../../glossary.md#term-scalar); [Shapes glossary](../../glossary.md#shape); [layout glossary](../../glossary.md#layout))</span>
- Stored values, including [signed zero](../../glossary.md#term-signed-zero), [infinity](../../glossary.md#term-infinity), and [NaN](../../glossary.md#term-nan), are not inspected during construction.
- Chaining a transform records the immediately preceding result as its exact input.
- Exhausting the factory's Tensor-ID space fails after local descriptor and provenance values are
  built.<span data-glossary-navigation="true"> ([descriptor glossary](../../glossary.md#term-descriptor); [provenance glossary](../../glossary.md#provenance))</span>

<a id="scalar-arithmetic-and-clamp-expressions"></a>
### [Scalar](../../glossary.md#term-scalar) arithmetic and clamp expressions

The scalar arithmetic surface parallels the seven [Tensor](../../glossary.md#tensor)-to-Tensor relationships. Each has one
exact [`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue) overload and one retained exact-[FLOAT64](../../glossary.md#term-float64) `double` convenience:

| Typed method | FLOAT64 convenience | [Elementwise](../../glossary.md#term-elementwise) meaning | Attributes |
|---|---|---|---|
| `add(ScalarValue)` | `add(double)` | Add the scalar. | `ScalarValueAttrs` |
| `sub(ScalarValue)` | `sub(double)` | Subtract the scalar from the input. | `ScalarValueAttrs` |
| `mul(ScalarValue)` | `mul(double)` | Multiply by the scalar. | `ScalarValueAttrs` |
| `div(ScalarValue)` | `div(double)` | Divide the input by the scalar. | `ScalarValueAttrs` |
| `minimum(ScalarValue)` | `minimum(double)` | Select the pairwise scalar minimum. | `ScalarValueAttrs` |
| `maximum(ScalarValue)` | `maximum(double)` | Select the pairwise scalar maximum. | `ScalarValueAttrs` |
| `pow(ScalarValue)` | `pow(double)` | Raise the input to the scalar [exponent](../../glossary.md#term-exponent). | `ScalarValueAttrs` |

The clamp surface retains one first-class range [operation](../../glossary.md#operation) and two one-bound conveniences:

| Typed method | FLOAT64 convenience | Meaning | Stored kind/attributes |
|---|---|---|---|
| `clamp(ScalarValue, ScalarValue)` | `clamp(double, double)` | Constrain to inclusive bounds. | `CLAMP` + `ClampRangeAttrs` |
| `clampMin(ScalarValue)` | `clampMin(double)` | Apply an inclusive lower bound through scalar maximum. | `MAX` + `ScalarValueAttrs` |
| `clampMax(ScalarValue)` | `clampMax(double)` | Apply an inclusive upper bound through scalar minimum. | `MIN` + `ScalarValueAttrs` |

Typed ADD, SUB, MUL, MIN, and MAX plus `clampMin` and `clampMax` accept floating or signed-integral
receivers. Typed DIV, POW, and first-class two-bound CLAMP remain floating-only. Every typed value
must have the receiver's exact [data type](../../glossary.md#data-type--datatype); scalar attributes never promote. The `double` methods
construct exact FLOAT64 values; they do not
infer or narrow to another receiver type. Thus `float32Input.mul(0.5d)` fails, while
`float32Input.mul(ScalarValue.float32(0.5f))` is valid. The result retains the exact input data
type and [immutable](../../glossary.md#term-immutable) [`Shape`](../../glossary.md#shape) reference, preserves `requiresGrad`, and leaves [layout](../../glossary.md#layout) unresolved.
Integral results necessarily have false [gradient](../../glossary.md#term-gradient) eligibility.

Every valid call returns a fresh Tensor with a new factory [identity](../../glossary.md#term-identity-meanings), no label, and no [host storage](../../glossary.md#host-storage).
Its [provenance](../../glossary.md#provenance) contains the exact matching `ScalarElementwiseKind`, typed attributes, and exactly
the receiver reference. `clamp(minValue, maxValue)` is one first-class `CLAMP` operation with the
exactly ordered value meaning `MIN(MAX(input, minValue), maxValue)`, but no stored intermediate
[producers](../../glossary.md#term-producer). Floating scalar [extrema](../../glossary.md#saturation-pole-and-extremum) compare represented numeric values: [NaN](../../glossary.md#term-nan) propagates, opposite
signed zeros select [negative zero](../../glossary.md#nan-infinity-and-signed-zero) for MIN or [positive zero](../../glossary.md#nan-infinity-and-signed-zero) for MAX, and other values use ordinary
numeric order. Exact `ScalarValue` bits remain [metadata](../../glossary.md#term-metadata) identity rather than bitwise comparison.
Consequently, a NaN input or bound makes CLAMP NaN; equal non-NaN same-representation bounds
produce that bound; bounds `[-0, +0]` select the corresponding directional zero; and bounds
`[+0, -0]` produce negative zero for every non-NaN input. `clampMin` creates exactly one scalar
`MAX` [producer](../../glossary.md#term-producer), and `clampMax` exactly one scalar `MIN` producer. Repeated, nested, identity-like,
and special-value calls are not interned, folded, or simplified at this public expression-
construction boundary.

[Compiler](../../glossary.md#term-compiler) first-order [AD](../../glossary.md#autograd) implements the eight exact-type floating scalar kinds, including
DIV, POW, MIN, MAX and CLAMP. Extrema split exact ties by one half; CLAMP applies that rule
to ordered MIN(MAX(x,min),max), including one quarter when tied at both stages. Scalar
attributes/bounds have no [cotangent](../../glossary.md#cotangent). The ordinary formula [owner](../../glossary.md#term-owner) preserves the selected
singularity, exceptional-value and endpoint behavior without a separate numerical policy.

The current package-private compiler has one narrower, guarded internal rewrite after [canonical](../../glossary.md#term-canonical-form)
[validation](../../glossary.md#term-validation). It may bypass scalar `MUL` by exact typed positive one for numeric types,
scalar `DIV` and `POW` by exact typed positive one for floating types, and scalar `ADD` and `SUB`
by exact typed zero for [`INT32`](../../glossary.md#data-type--datatype) and [`INT64`](../../glossary.md#data-type--datatype). It also recognizes duplicate-input [binary](../../glossary.md#term-binary-descriptor-and-metadata) `MIN` and
`MAX`. Each bypass is limited to a non-gradient, one-output internal [occurrence](../../glossary.md#term-occurrence) in either [graph
phase](../../glossary.md#graph-phase) with complete input/output [descriptor](../../glossary.md#term-descriptor) equality and a result that is not a [graph](../../glossary.md#graph) output.
The occurrence phase is preserved. This does not change the public Tensor's fresh identity or
provenance construction: the compiler consumes the later immutable graph occurrence.

For the scalar rows, `ScalarValueAttrs` is immutable semantic metadata, not a Tensor [constant](../../glossary.md#term-constant).
The compiler reads only its exact type and typed value; it does not inspect Tensor host storage or
expose a public compiler [API](../../glossary.md#term-application-programming-interface-api). A later current [scan](../../glossary.md#term-cumulative-scan) can combine an explicit integral logical-splat
fact for the graph-valued input with an exact same-typed integral attribute for scalar `ADD`,
`SUB`, `MUL`, `MIN`, or `MAX`. That result is a new fixed [logical splat](../../glossary.md#logical-splat), not a Tensor, dense
[payload](../../glossary.md#artifact-envelope-and-atomic-publication), or physical allocation. Floating scalar [evaluation](../../glossary.md#term-evaluation), CLAMP, DIV, POW, multiplication by
zero as an identity rewrite, [cancellation](../../glossary.md#term-cancellation), other exponents, bounds/clamp identities, and broader
algebra remain unsupported compiler rewrites.

Factory-created [leaves](../../glossary.md#term-leaf) are not constants merely because they have host storage or were created by
`scalar`, `zeros`, `ones`, `full`, import, allocation, or random construction. The package-private
compiler recognizes a fixed leaf only through explicit exact Tensor-identity ingress. An absent
leaf remains bindable, and changing, replacing, clearing, or losing its [host](../../glossary.md#term-host)-storage association
does not create or alter a supplied immutable splat fact.

Floating scalar and Tensor-to-Tensor extrema share the same selected semantics: NaN propagates,
[infinities](../../glossary.md#nan-infinity-and-signed-zero)
use normal numerical order, minimum selects negative zero from opposite signed zeros, and maximum
selects positive zero. The [model](../../glossary.md#term-model) promises no [NaN payload](../../glossary.md#nan-infinity-and-signed-zero), gradient convention, algorithm, [backend
route](../../glossary.md#backend-route), or execution. Integral MIN/MAX use ordinary signed order, and integral scalar ADD/SUB/MUL
use fixed-width [two's-complement](../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) modular semantics. Expression construction records but does not
evaluate those meanings.

For `clamp`, null checks and floating-input eligibility precede bound construction. The bounds
must share one non-[BOOL](../../glossary.md#data-type--datatype) type, and `ClampRangeAttrs` compares the exact represented primitive type.
It rejects only a strict inversion; equal bounds, either ordering of signed zeros, ordered
infinities, and a floating NaN endpoint remain representable. The common bound type must then
equal the receiver type. NaN payload selection, numerical evaluation, optimization, gradients,
[graph capture](../../glossary.md#capture-graph-ports-and-reconstruction), and [backend](../../glossary.md#backend) execution remain later responsibilities.

<a id="complete-scalar-expression-example"></a>
#### Complete [scalar](../../glossary.md#term-scalar)-expression example

<a id="goal-and-inputs-19"></a>
##### Goal and inputs

Build a [scalar](../../glossary.md#term-scalar) multiplication followed by one range-clamp expression from a storage-free
[`FLOAT32`](../../glossary.md#term-float32) [tensor](../../glossary.md#tensor) of [shape](../../glossary.md#shape) `[2, 3]`. The example uses [negative zero](../../glossary.md#nan-infinity-and-signed-zero) and a raw [NaN payload](../../glossary.md#nan-infinity-and-signed-zero) to
demonstrate exact [binary32](../../glossary.md#floating-point-precision-and-rounding) retention and observes expression [metadata](../../glossary.md#term-metadata) rather than numerical
values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class ScalarExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));

        ScalarValue negativeZero = ScalarValue.float32(-0.0f);
        ScalarValue payloadNaN = ScalarValue.float32(
                Float.intBitsToFloat(0x7FC0_0042));
        Tensor scaled = input.mul(negativeZero);
        Tensor result = scaled.clamp(negativeZero, payloadNaN);
        TensorProvenance scaledFrom = scaled.provenance().orElseThrow();
        TensorProvenance clampedFrom = result.provenance().orElseThrow();
        ScalarValueAttrs multiplier =
                (ScalarValueAttrs) scaledFrom.operation().attrs();
        ClampRangeAttrs range =
                (ClampRangeAttrs) clampedFrom.operation().attrs();

        System.out.println("type=" + result.descriptor().dataType());
        System.out.println("sameShape=" + (result.descriptor().shape() == shape));
        System.out.println("layoutUnresolved=" + result.descriptor().layout().isEmpty());
        System.out.println("requiresGrad=" + result.descriptor().requiresGrad());
        System.out.println("unlabeled=" + result.label().isEmpty());
        System.out.println("storageFree=" + result.hostStorage().isEmpty());
        System.out.println("multiplierKind=" + scaledFrom.operation().kind());
        System.out.println("multiplierBits="
                + Integer.toHexString(
                        Float.floatToRawIntBits(multiplier.value().float32Value())));
        System.out.println("clampKind=" + clampedFrom.operation().kind());
        System.out.println("minBits="
                + Integer.toHexString(
                        Float.floatToRawIntBits(range.minValue().float32Value())));
        System.out.println("maxBits="
                + Integer.toHexString(
                        Float.floatToRawIntBits(range.maxValue().float32Value())));
        System.out.println("exactInput=" + (clampedFrom.inputs().getFirst() == scaled));
        System.out.println("oneClampInput=" + (clampedFrom.inputs().size() == 1));
        System.out.println("fresh=" + (result != scaled && scaled != input));
    }
}
```

<a id="meaningful-lines-and-intermediate-results-5"></a>
##### Meaningful lines and intermediate results

- `input.mul(negativeZero)` records [scalar](../../glossary.md#term-scalar) `MUL` and retains the exact [`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue) reference and
  binary32 negative-zero pattern in one <span data-glossary-navigation="true"> ([binary32 glossary](../../glossary.md#floating-point-precision-and-rounding))</span>`ScalarValueAttrs`.
- `scaled.clamp(negativeZero, payloadNaN)` records exactly one `CLAMP` with both exact typed values
  in one `ClampRangeAttrs`. Its sole input
  is the exact `scaled` object, so the chain preserves immediate expression [provenance](../../glossary.md#provenance).
- Both calls preserve [`FLOAT32`](../../glossary.md#term-float32), the exact [`shape`](../../glossary.md#shape) reference, and true gradient eligibility while
  creating fresh unresolved, unlabeled, storage-free results.<span data-glossary-navigation="true"> ([gradient glossary](../../glossary.md#term-gradient))</span>

<a id="result-and-interpretation-19"></a>
##### Result and interpretation

The program prints:

```text
type=FLOAT32
sameShape=true
layoutUnresolved=true
requiresGrad=true
unlabeled=true
storageFree=true
multiplierKind=MUL
multiplierBits=80000000
clampKind=CLAMP
minBits=80000000
maxBits=7fc00042
exactInput=true
oneClampInput=true
fresh=true
```

The output proves exact attribute-bit retention, first-class range-clamp [identity](../../glossary.md#term-identity-meanings), [descriptor](../../glossary.md#term-descriptor)
retention, fresh identity, and one-input [provenance](../../glossary.md#provenance). It does not prove multiplication or clamp
values, [parameter](../../glossary.md#term-parameter) conversion for a [backend](../../glossary.md#backend), [gradient](../../glossary.md#term-gradient) rules, [graph capture](../../glossary.md#capture-graph-ports-and-reconstruction), backend support, or
execution.

<a id="failures-and-useful-variations-17"></a>
##### Failures and useful variations

- Exact typed integral ADD, SUB, MUL, MIN, MAX, `clampMin`, and `clampMax` are valid. Integral DIV,
  POW, and range CLAMP fail before result identity allocation; BOOL remains invalid.<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings); [BOOL glossary](../../glossary.md#data-type--datatype))</span>
- `clamp(ScalarValue.float32(2.0f), ScalarValue.float32(1.0f))` fails with
  `IllegalArgumentException` and message
  `minValue must be less than or equal to maxValue`. On a non-floating input, the data-type failure
  occurs before that range failure.
- Signed zeros, infinities, and NaN payload bits remain unchanged in attributes. A NaN clamp
  endpoint is accepted because primitive <span data-glossary-navigation="true"> ([infinities glossary](../../glossary.md#nan-infinity-and-signed-zero); [NaN payload glossary](../../glossary.md#nan-infinity-and-signed-zero))</span>`>` is false when either operand is [NaN](../../glossary.md#term-nan).
- A `double` convenience used on FLOAT32, BFLOAT16, INT32, or INT64 fails exact data-type matching;
  callers must construct the matching typed value explicitly.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../glossary.md#term-float32); [BFLOAT16 glossary](../../glossary.md#term-bfloat16); [INT32 glossary](../../glossary.md#data-type--datatype); [INT64 glossary](../../glossary.md#data-type--datatype))</span>
- Scalar, zero-sized, ordinary static, and dynamic shapes are accepted. Repeated calls and values
  such as multiplier <span data-glossary-navigation="true"> ([Scalar glossary](../../glossary.md#term-scalar); [shapes glossary](../../glossary.md#shape))</span>`1.0` still create fresh expressions without [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form).


---

[Previous topic](selection-casts.md) · [Next topic](reductions.md)
