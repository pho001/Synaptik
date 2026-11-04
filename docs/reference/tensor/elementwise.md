# Unary transforms, scalar arithmetic and clamps

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**Reader model.** **Unary** means one input; an **elementwise** function transforms each
position without mixing it with neighboring positions. A **scalar** parameter supplies one
same-typed value for all positions. A **clamp** limits values to a range.
An **activation** is a numeric transform used between learned computations: ReLU keeps positive
values and maps negative values to zero, while sigmoid smoothly maps real values toward the
interval from zero to one. A floating **classification** instead returns a true/false tensor
identifying finite values, NaNs, or infinities; it is not a machine-learning class prediction.
The exact functions and exceptional-value rules remain in the tables below.

For the function names below, a **reciprocal** is `1/x`; **exp** is the natural exponential
with base `e`, approximately `2.71828`; and the natural **logarithm** is its inverse.
Thus `exp(0) = 1` and `log(1) = 0` in exact mathematics. **Tanh** smoothly maps real inputs
between negative one and one. The Gaussian error function **erf** is related to the probability
of values falling within a centered normal distribution; GELU's `Phi` is the cumulative
probability up to its argument for a mean-zero, unit-standard-deviation normal distribution.
These explanations orient the reader; the formulas and special-value rules below remain exact.

<a id="unary-numeric-transforms-and-floating-classifications"></a>
### Unary numeric transforms and floating classifications

Nineteen zero-argument unary methods create one-input, floating-preserving elementwise semantics:

| Method | Elementwise meaning |
|---|---|
| `abs` | Absolute magnitude. |
| `neg` | Additive inverse. |
| `reciprocal` | Multiplicative reciprocal. |
| `log` | Natural logarithm. |
| `log1p` | Natural logarithm of one plus the input. |
| `exp` | Portable natural exponential request. |
| `expm1` | Natural exponential of the input minus one. |
| `erf` | Gaussian error function. |
| `sqrt` | Principal square root. |
| `rsqrt` | Reciprocal of the principal square root. |
| `floor` | Greatest integer-valued result not greater than the input. |
| `ceil` | Least integer-valued result not less than the input. |
| `sign` | Numeric negative, zero, or positive classification. |
| `relu` | Rectified linear unit. |
| `sigmoid` | Logistic sigmoid. |
| `tanh` | Portable hyperbolic tangent request. |
| `gelu` | Exact Gaussian error linear unit, `x * Phi(x)`. |
| `geluTanhApproximation` | Fixed conventional tanh approximation to GELU. |
| `silu` | Sigmoid linear unit, `x * sigmoid(x)`. |

Each method accepts every floating type, including both FP8 types. The result retains the exact
input data type, immutable `Shape` reference, and `requiresGrad` flag, while leaving layout unresolved.
That flag is eligibility metadata; it does not assert that a derivative or backward rule exists.

`rsqrt`, `log1p`, `expm1`, both GELU variants, and SiLU are first-class transforms, not stored
compositions. Model owns one family/dtype result contract for these and the other unary kinds.
The six exact represented-value kinds `ABS`, `NEG`, `SIGN`, `FLOOR`, `CEIL`, and `RELU` have no
arithmetic DAZ/FTZ permission. Other named FLOAT32 and low arithmetic primitive inputs
may use denormals-are-zero (DAZ), and results may use flush-to-zero (FTZ); FLOAT64 forbids both.
Each kind retains its mathematical function, domain, special class, range, and required zero-sign
rules. Finite thresholds are route-qualified in tests, not public ordered-distance or forward-
error envelopes. Composite `RSQRT`, sigmoid, GELU, tanh GELU, and SiLU inherit only their named
primitive sites and exact guards; no final-output tolerance applies.

Exact GELU uses `0.5 * x * (1 + erf(x / sqrt(2)))`. Its constants, square root, division, `ERF`,
addition, and two multiplications are its complete sites. The explicitly named tanh approximation
uses `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`; its constants, `x*x`, `x^2*x`,
root, division, additions, remaining multiplications, and `TANH` are complete. Sigmoid uses
`1/(1+exp(-x))` for nonnegative input and `exp(x)/(1+exp(x))` for negative input. SiLU uses
`x/(1+exp(-x))` and `x*exp(x)/(1+exp(x))` on the corresponding branches. Comparison, negation,
exponential, additions, multiplication, and divisions are explicit sites. The API uses canonical
`silu` naming and provides no `swish` alias. None of these first-class operations gains a
whole-result tolerance envelope.

Their Model-owned special-value meanings are:

| Method | Special-value behavior |
|---|---|
| `rsqrt` | Positive/negative zero becomes same-signed infinity; positive infinity becomes positive zero; negative finite values and negative infinity produce NaN; NaN produces NaN. |
| `log1p` | Signed zero is preserved; `-1` produces negative infinity; values below `-1`, including negative infinity, produce NaN; positive infinity remains positive infinity; NaN produces NaN. |
| `expm1` | Signed zero is preserved; negative infinity becomes exactly `-1`; positive infinity remains positive infinity; NaN produces NaN. |
| `gelu` | Negative infinity becomes negative zero by continuous extension; signed zero is preserved; positive infinity remains positive infinity; NaN produces NaN. |
| `geluTanhApproximation` | Negative infinity becomes negative zero by continuous extension; signed zero is preserved; positive infinity remains positive infinity; NaN produces NaN. |
| `silu` | Negative infinity becomes negative zero by continuous extension; signed zero is preserved; positive infinity remains positive infinity; NaN produces NaN. |

These meanings specify results, not Tensor construction-time evaluation or a backend capability.
Compiler and backend work may preserve useful near-zero accuracy for `log1p` and `expm1` because
the model records the transforms without decomposition.

Three separate floating-classification methods describe BOOL values:

| Method | True exactly when |
|---|---|
| `isFinite` | The input is finite: normal, subnormal, positive zero, or negative zero. |
| `isNaN` | The input is NaN, independent of sign, quiet/signaling encoding, or payload. |
| `isInf` | The input is positive or negative infinity. |

Classification accepts the same four floating input types but always produces `BOOL`, retains the
exact input `Shape`, leaves layout unresolved, and sets `requiresGrad=false`. These methods
are non-differentiable value classifications, not numeric unary transforms, trace diagnostics,
eager Java booleans, or validation checks. For every represented floating value, exactly one of
the three classifications is true when the operation is eventually evaluated.

Every valid transform or classification call returns a fresh Tensor with no label or host
storage. Its provenance contains one parameterless `Operation`, exactly the receiver reference,
one producer output, and output index zero. Model construction reads no values and calculates no
result.

<a id="complete-transform-and-classification-example"></a>
#### Complete transform-and-classification example

<a id="goal-and-inputs-18"></a>
##### Goal and inputs

Build one exact `GELU` transform and one `IS_NAN` classification from a storage-free `FLOAT32`
Tensor of Shape `[2, 3]`. The example observes metadata only; it does not claim transformed or
classified element values.

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
  floating type and gradient-eligibility request.
- `input.isNaN()` records one distinct `FloatingClassificationKind.IS_NAN` occurrence and fixes
  the result descriptor to non-gradient `BOOL`.
- Both results retain the exact `shape` and input references, use canonical no-attributes state,
  and select output index zero from a one-output producer.

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

The output proves the result-type distinction, exact operation kinds, Shape and input retention,
parameterless construction, and one-input, output-index-zero provenance. It does not prove that
GELU or NaN classification ran, that numerical accuracy or a gradient rule exists, that a graph
was captured, or that any backend supports either operation.

<a id="failures-and-useful-variations-16"></a>
##### Failures and useful variations

- Calling any of these methods on an `INT32`, `INT64`, or `BOOL` Tensor fails with
  `IllegalArgumentException` before result identity allocation; no implicit cast is inserted.
- Scalar, zero-sized, ordinary static, and dynamic Shapes are accepted. A resolved input layout is
  deliberately not copied to either result.
- Stored values, including signed zero, infinity, and NaN, are not inspected during construction.
- Chaining a transform records the immediately preceding result as its exact input.
- Exhausting the factory's Tensor-ID space fails after local descriptor and provenance values are
  built.

<a id="scalar-arithmetic-and-clamp-expressions"></a>
### Scalar arithmetic and clamp expressions

The scalar arithmetic surface parallels the seven Tensor-to-Tensor relationships. Each has one
exact `ScalarValue` overload and one retained exact-FLOAT64 `double` convenience:

| Typed method | FLOAT64 convenience | Elementwise meaning | Attributes |
|---|---|---|---|
| `add(ScalarValue)` | `add(double)` | Add the scalar. | `ScalarValueAttrs` |
| `sub(ScalarValue)` | `sub(double)` | Subtract the scalar from the input. | `ScalarValueAttrs` |
| `mul(ScalarValue)` | `mul(double)` | Multiply by the scalar. | `ScalarValueAttrs` |
| `div(ScalarValue)` | `div(double)` | Divide the input by the scalar. | `ScalarValueAttrs` |
| `minimum(ScalarValue)` | `minimum(double)` | Select the pairwise scalar minimum. | `ScalarValueAttrs` |
| `maximum(ScalarValue)` | `maximum(double)` | Select the pairwise scalar maximum. | `ScalarValueAttrs` |
| `pow(ScalarValue)` | `pow(double)` | Raise the input to the scalar exponent. | `ScalarValueAttrs` |

The clamp surface retains one first-class range operation and two one-bound conveniences:

| Typed method | FLOAT64 convenience | Meaning | Stored kind/attributes |
|---|---|---|---|
| `clamp(ScalarValue, ScalarValue)` | `clamp(double, double)` | Constrain to inclusive bounds. | `CLAMP` + `ClampRangeAttrs` |
| `clampMin(ScalarValue)` | `clampMin(double)` | Apply an inclusive lower bound through scalar maximum. | `MAX` + `ScalarValueAttrs` |
| `clampMax(ScalarValue)` | `clampMax(double)` | Apply an inclusive upper bound through scalar minimum. | `MIN` + `ScalarValueAttrs` |

Typed ADD, SUB, MUL, MIN, and MAX plus `clampMin` and `clampMax` accept floating or signed-integral
receivers. Typed DIV, POW, and first-class two-bound CLAMP remain floating-only. Every typed value
must have the receiver's exact data type; scalar attributes never promote. The `double` methods
construct exact FLOAT64 values; they do not
infer or narrow to another receiver type. Thus `float32Input.mul(0.5d)` fails, while
`float32Input.mul(ScalarValue.float32(0.5f))` is valid. The result retains the exact input data
type and immutable `Shape` reference, preserves `requiresGrad`, and leaves layout unresolved.
Integral results necessarily have false gradient eligibility.

Every valid call returns a fresh Tensor with a new factory identity, no label, and no host storage.
Its provenance contains the exact matching `ScalarElementwiseKind`, typed attributes, and exactly
the receiver reference. `clamp(minValue, maxValue)` is one first-class `CLAMP` operation with the
exactly ordered value meaning `MIN(MAX(input, minValue), maxValue)`, but no stored intermediate
producers. Floating scalar extrema compare represented numeric values: NaN propagates, opposite
signed zeros select negative zero for MIN or positive zero for MAX, and other values use ordinary
numeric order. Exact `ScalarValue` bits remain metadata identity rather than bitwise comparison.
Consequently, a NaN input or bound makes CLAMP NaN; equal non-NaN same-representation bounds
produce that bound; bounds `[-0, +0]` select the corresponding directional zero; and bounds
`[+0, -0]` produce negative zero for every non-NaN input. `clampMin` creates exactly one scalar
`MAX` producer, and `clampMax` exactly one scalar `MIN` producer. Repeated, nested, identity-like,
and special-value calls are not interned, folded, or simplified at this public expression-
construction boundary.

Compiler first-order AD implements the eight exact-type floating scalar kinds, including
DIV, POW, MIN, MAX and CLAMP. Extrema split exact ties by one half; CLAMP applies that rule
to ordered MIN(MAX(x,min),max), including one quarter when tied at both stages. Scalar
attributes/bounds have no cotangent. The ordinary formula owner preserves the selected
singularity, exceptional-value and endpoint behavior without a separate numerical policy.

The current package-private compiler has one narrower, guarded internal rewrite after canonical
validation. It may bypass scalar `MUL` by exact typed positive one for numeric types,
scalar `DIV` and `POW` by exact typed positive one for floating types, and scalar `ADD` and `SUB`
by exact typed zero for `INT32` and `INT64`. It also recognizes duplicate-input binary `MIN` and
`MAX`. Each bypass is limited to a non-gradient, one-output internal occurrence in either graph
phase with complete input/output descriptor equality and a result that is not a graph output.
The occurrence phase is preserved. This does not change the public Tensor's fresh identity or
provenance construction: the compiler consumes the later immutable graph occurrence.

For the scalar rows, `ScalarValueAttrs` is immutable semantic metadata, not a Tensor constant.
The compiler reads only its exact type and typed value; it does not inspect Tensor host storage or
expose a public compiler API. A later current scan can combine an explicit integral logical-splat
fact for the graph-valued input with an exact same-typed integral attribute for scalar `ADD`,
`SUB`, `MUL`, `MIN`, or `MAX`. That result is a new fixed logical splat, not a Tensor, dense
payload, or physical allocation. Floating scalar evaluation, CLAMP, DIV, POW, multiplication by
zero as an identity rewrite, cancellation, other exponents, bounds/clamp identities, and broader
algebra remain unsupported compiler rewrites.

Factory-created leaves are not constants merely because they have host storage or were created by
`scalar`, `zeros`, `ones`, `full`, import, allocation, or random construction. The package-private
compiler recognizes a fixed leaf only through explicit exact Tensor-identity ingress. An absent
leaf remains bindable, and changing, replacing, clearing, or losing its host-storage association
does not create or alter a supplied immutable splat fact.

Floating scalar and Tensor-to-Tensor extrema share the same selected semantics: NaN propagates,
infinities
use normal numerical order, minimum selects negative zero from opposite signed zeros, and maximum
selects positive zero. The model promises no NaN payload, gradient convention, algorithm, backend
route, or execution. Integral MIN/MAX use ordinary signed order, and integral scalar ADD/SUB/MUL
use fixed-width two's-complement modular semantics. Expression construction records but does not
evaluate those meanings.

For `clamp`, null checks and floating-input eligibility precede bound construction. The bounds
must share one non-BOOL type, and `ClampRangeAttrs` compares the exact represented primitive type.
It rejects only a strict inversion; equal bounds, either ordering of signed zeros, ordered
infinities, and a floating NaN endpoint remain representable. The common bound type must then
equal the receiver type. NaN payload selection, numerical evaluation, optimization, gradients,
graph capture, and backend execution remain later responsibilities.

<a id="complete-scalar-expression-example"></a>
#### Complete scalar-expression example

<a id="goal-and-inputs-19"></a>
##### Goal and inputs

Build a scalar multiplication followed by one range-clamp expression from a storage-free
`FLOAT32` tensor of shape `[2, 3]`. The example uses negative zero and a raw NaN payload to
demonstrate exact binary32 retention and observes expression metadata rather than numerical
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

- `input.mul(negativeZero)` records scalar `MUL` and retains the exact `ScalarValue` reference and
  binary32 negative-zero pattern in one `ScalarValueAttrs`.
- `scaled.clamp(negativeZero, payloadNaN)` records exactly one `CLAMP` with both exact typed values
  in one `ClampRangeAttrs`. Its sole input
  is the exact `scaled` object, so the chain preserves immediate expression provenance.
- Both calls preserve `FLOAT32`, the exact `shape` reference, and true gradient eligibility while
  creating fresh unresolved, unlabeled, storage-free results.

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

The output proves exact attribute-bit retention, first-class range-clamp identity, descriptor
retention, fresh identity, and one-input provenance. It does not prove multiplication or clamp
values, parameter conversion for a backend, gradient rules, graph capture, backend support, or
execution.

<a id="failures-and-useful-variations-17"></a>
##### Failures and useful variations

- Exact typed integral ADD, SUB, MUL, MIN, MAX, `clampMin`, and `clampMax` are valid. Integral DIV,
  POW, and range CLAMP fail before result identity allocation; BOOL remains invalid.
- `clamp(ScalarValue.float32(2.0f), ScalarValue.float32(1.0f))` fails with
  `IllegalArgumentException` and message
  `minValue must be less than or equal to maxValue`. On a non-floating input, the data-type failure
  occurs before that range failure.
- Signed zeros, infinities, and NaN payload bits remain unchanged in attributes. A NaN clamp
  endpoint is accepted because primitive `>` is false when either operand is NaN.
- A `double` convenience used on FLOAT32, BFLOAT16, INT32, or INT64 fails exact data-type matching;
  callers must construct the matching typed value explicitly.
- Scalar, zero-sized, ordinary static, and dynamic shapes are accepted. Repeated calls and values
  such as multiplier `1.0` still create fresh expressions without canonicalization.


---

[Previous topic](selection-casts.md) · [Next topic](reductions.md)
