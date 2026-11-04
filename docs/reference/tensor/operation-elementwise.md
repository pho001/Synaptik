# [Unary](../../glossary.md#term-unary), classification and [scalar](../../glossary.md#term-scalar) semantic kinds

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**[Reader model](../../glossary.md#term-reader-model).** **Unary** operations have one input, and **[elementwise](../../glossary.md#term-elementwise)** operations act at each
logical position without mixing positions. A numeric transform returns a number; a [floating
classification](../../glossary.md#floating-classification) returns a true/false answer such as “is this [NaN](../../glossary.md#term-nan)?” A scalar [operation](../../glossary.md#operation) uses one
fixed typed [parameter](../../glossary.md#term-parameter) at every position. That parameter is an **attribute**, not another [Tensor](../../glossary.md#tensor)
input. **First-class** means the operation is recorded directly under its own kind rather than
as a stored chain of simpler operations. The [Tensor transform page](elementwise.md) explains
the functions and input-aware construction rules; these kinds only name their meanings.

<a id="unary-elementwise-semantic-kinds"></a>
### [Unary](../../glossary.md#term-unary) [elementwise](../../glossary.md#term-elementwise) semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly these [constants](../../glossary.md#term-constant) in declaration order:

| Kind | Elementwise meaning |
|---|---|
| `ABS` | Absolute magnitude of the input value. |
| `NEG` | Additive inverse of the input value. |
| `RECIPROCAL` | Multiplicative reciprocal of the input value. |
| [`LOG`](../../glossary.md#term-log) | [Natural logarithm](../../glossary.md#norms-elementary-functions-and-probability-distributions) of the input value. |
| `LOG1P` | Natural logarithm of one plus the input value. |
| [`EXP`](../../glossary.md#term-expx) | [Natural exponential](../../glossary.md#norms-elementary-functions-and-probability-distributions) of the input value. |
| `EXPM1` | Natural exponential of the input value minus one. |
| [`ERF`](../../glossary.md#term-erf) | [Gaussian error function](../../glossary.md#norms-elementary-functions-and-probability-distributions) of the input value. |
| `SQRT` | Principal [square root](../../glossary.md#norms-elementary-functions-and-probability-distributions) of the input value. |
| `RSQRT` | Reciprocal of the principal square root of the input value. |
| `FLOOR` | Greatest integer-valued result not greater than the input value. |
| `CEIL` | Least integer-valued result not less than the input value. |
| `SIGN` | Negative, zero, or positive sign classification represented numerically. |
| `RELU` | [Rectified linear unit](../../glossary.md#bias-and-activation) of the input value. |
| [`SIGMOID`](../../glossary.md#term-sigmoid) | Logistic sigmoid of the input value. |
| [`TANH`](../../glossary.md#term-tanh) | [Hyperbolic tangent](../../glossary.md#bias-and-activation) of the input value. |
| [`GELU`](../../glossary.md#gaussian-error-linear-unit--gelu) | Exact [Gaussian error linear unit](../../glossary.md#gaussian-error-linear-unit--gelu), `x * Phi(x)`. |
| `GELU_TANH_APPROXIMATION` | Fixed conventional tanh GELU approximation with coefficient `0.044715`. |
| `SILU` | [Sigmoid linear unit](../../glossary.md#sigmoid-linear-unit--silu), `x * sigmoid(x)`. |

All nineteen kinds have one logical input and no intrinsic [parameters](../../glossary.md#term-parameter). Their shared [signature](../../glossary.md#term-signature)
declares exact [`NoOperationAttrs`](../../glossary.md#nooperationattrs), one input, and one output; [canonical](../../glossary.md#term-canonical-form) composition remains
explicit:

```java
Operation exponential = new Operation(
        UnaryElementwiseKind.EXP,
        NoOperationAttrs.INSTANCE);
```

The enum does not store the input, infer a result [descriptor](../../glossary.md#term-descriptor), or create [provenance](../../glossary.md#provenance). The current
public unary [Tensor](../../glossary.md#tensor) methods own those expression-construction rules. `RSQRT`, `LOG1P`, `EXPM1`,
`GELU`, `GELU_TANH_APPROXIMATION`, and `SILU` remain first-class operations rather than [graph](../../glossary.md#graph)
decompositions. The two GELU kinds select distinct fixed functions; neither carries configurable
attributes. Every kind now has an explicit strict result rule through the selected represented,
Java elementary, or fixed-formula realization, together with exact domain, special-class, and
[signed-zero](../../glossary.md#nan-infinity-and-signed-zero) requirements. The kind itself still selects no [gradient](../../glossary.md#term-gradient) rule or [backend route](../../glossary.md#backend-route).
Selected formulas and special-value semantics are documented in [Unary numeric transforms and
floating classifications](elementwise.md#unary-numeric-transforms-and-floating-classifications).
Inherited enum names are diagnostic text, not [serialization](../../glossary.md#term-serialization) or [dispatch](../../glossary.md#term-dispatch) keys, and equally named
kinds from another family remain different typed values.

<a id="floating-classification-semantic-kinds"></a>
### Floating-classification semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind`
implements [`OperationKind`](../../glossary.md#operationkind) with exactly `IS_FINITE`, `IS_NAN`, and `IS_INF`, in that order. Each
kind has exact [`NoOperationAttrs`](../../glossary.md#nooperationattrs), one input, and one output. The semantic family is separate from
`UnaryElementwiseKind` because its public construction produces fixed non-[differentiable](../../glossary.md#derivative-and-chain-rule) [BOOL](../../glossary.md#data-type--datatype)
[metadata](../../glossary.md#term-metadata) instead of preserving a floating result type.

The kind values name [elementwise](../../glossary.md#term-elementwise) classifications only. They do not retain an operand [descriptor](../../glossary.md#term-descriptor),
infer BOOL metadata, inspect a [NaN](../../glossary.md#term-nan) encoding or other stored value, create [provenance](../../glossary.md#provenance), define
[gradients](../../glossary.md#term-gradient), execute classification, or report [backend](../../glossary.md#backend) support. The public [Tensor](../../glossary.md#tensor) methods own
floating-only input [validation](../../glossary.md#term-validation), exact [Shape](../../glossary.md#shape) retention, unresolved [layout](../../glossary.md#layout), false gradient
eligibility, and one-input provenance. “Classification” here is [graph](../../glossary.md#graph)-visible result semantics;
it is unrelated to tracing diagnostics.

<a id="scalar-arithmetic-and-clamp-semantic-kinds"></a>
### [Scalar](../../glossary.md#term-scalar) arithmetic and clamp semantic kinds

The public enum
`io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind` implements
[`OperationKind`](../../glossary.md#operationkind) with exactly these [constants](../../glossary.md#term-constant) in declaration order:

| Kind | [Elementwise](../../glossary.md#term-elementwise) meaning | Required attributes |
|---|---|---|
| `ADD` | Input value plus a scalar addend. | `ScalarValueAttrs` |
| `SUB` | Input value minus a scalar subtrahend. | `ScalarValueAttrs` |
| `MUL` | Input value multiplied by a scalar multiplier. | `ScalarValueAttrs` |
| `DIV` | Input value divided by a scalar denominator. | `ScalarValueAttrs` |
| `MIN` | Pairwise minimum of the input and scalar candidate. | `ScalarValueAttrs` |
| `MAX` | Pairwise maximum of the input and scalar candidate. | `ScalarValueAttrs` |
| `POW` | Input value as the base raised to a scalar [exponent](../../glossary.md#term-exponent). | `ScalarValueAttrs` |
| `CLAMP` | Input value constrained to inclusive lower and upper scalar bounds. | `ClampRangeAttrs` |

The scalar [parameter](../../glossary.md#term-parameter) or bounds are [operation](../../glossary.md#operation) attributes, not additional [Tensor](../../glossary.md#tensor) inputs. The family
[signatures](../../glossary.md#term-signature) enforce the table: `CLAMP` accepts exactly `ClampRangeAttrs`, while the other seven kinds
accept exactly `ScalarValueAttrs`; every variant declares one input and one output.

`ScalarValueAttrs` retains one non-null [`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue) by exact reference without conversion,
[normalization](../../glossary.md#term-normalization), or input compatibility checks. `ClampRangeAttrs` requires two non-null values with
the same non-[BOOL](../../glossary.md#data-type--datatype) [data type](../../glossary.md#data-type--datatype) and compares that represented primitive type without a [binary64](../../glossary.md#floating-point-precision-and-rounding)
intermediate. It accepts equal bounds, either ordering of signed zeros, ordered [infinities](../../glossary.md#nan-infinity-and-signed-zero), and
any floating range containing a [NaN](../../glossary.md#term-nan) endpoint. An inverted range fails with
`IllegalArgumentException` and the message
`minValue must be less than or equal to maxValue`.

The following example creates semantic [descriptors](../../glossary.md#term-descriptor) directly; it does not call the matching public
Tensor expression methods or execute an operation:

```java
Operation scaled = new Operation(
        ScalarElementwiseKind.MUL,
        new ScalarValueAttrs(ScalarValue.float32(-0.0f)));

Operation clipped = new Operation(
        ScalarElementwiseKind.CLAMP,
        new ClampRangeAttrs(ScalarValue.float32(0.0f), ScalarValue.float32(1.0f)));
```

Both attribute records compose `ScalarValue` equality, which includes the exact data type and [raw
bits](../../glossary.md#storage-mapping-contiguity-and-injectivity). Positive and negative floating zero are unequal, and distinct NaN payloads remain unequal.
An `Operation` has no operand descriptor, so direct construction does not prove that these [FLOAT32](../../glossary.md#term-float32)
attributes will eventually be attached to a FLOAT32 input.

Scalar MIN/MAX use the same NaN propagation, normal [infinity](../../glossary.md#term-infinity) ordering, and opposite-zero choices
as the [binary](../../glossary.md#term-binary-descriptor-and-metadata) family. The input's represented numeric value is compared with the exact represented
value of the same-typed `ScalarValue`; the scalar's retained raw bits remain [metadata](../../glossary.md#term-metadata) equality and
do not make elementwise comparison bitwise. [FLOAT64](../../glossary.md#term-float64) and FLOAT32 use their represented [IEEE](../../glossary.md#term-ieee)-754
values, while [BFLOAT16](../../glossary.md#term-bfloat16) uses the value represented by its exact 16-bit storage. `CLAMP(input,
minValue, maxValue)` has the exactly ordered value meaning
`MIN(MAX(input, minValue), maxValue)` but remains one semantic [occurrence](../../glossary.md#term-occurrence) with no stored
intermediate [producers](../../glossary.md#term-producer):

| Input or bounds | Result |
|---|---|
| Input, `minValue`, or `maxValue` is NaN | NaN, with no [payload](../../glossary.md#artifact-envelope-and-atomic-publication) promise |
| Equal non-NaN, same-representation bounds | That bound for every non-NaN input |
| Bounds `[-0, +0]` | Negative inputs and [negative zero](../../glossary.md#nan-infinity-and-signed-zero) produce `-0`; positive inputs and [positive zero](../../glossary.md#nan-infinity-and-signed-zero) produce `+0` |
| Bounds `[+0, -0]` | Every non-NaN input produces `-0` |

These results add no [validation](../../glossary.md#term-validation): primitive represented-value `>` remains false for either
[signed-zero](../../glossary.md#nan-infinity-and-signed-zero) order and for NaN endpoints. The one-bound public methods are conveniences over scalar
MAX and MIN rather than separate kinds.

The kinds and attributes alone do not infer result descriptors, define scalar conversion, evaluate
the fixed numerical meaning, create [provenance](../../glossary.md#provenance), define [gradients](../../glossary.md#term-gradient), report [backend](../../glossary.md#backend) support, select
[kernels](../../glossary.md#kernel), or execute computation. Their enum names and generated record text are diagnostics, not
[serialization](../../glossary.md#term-serialization) or [dispatch](../../glossary.md#term-dispatch) contracts. The current public scalar Tensor methods separately own
floating-input validation, descriptor derivation, exact attribute composition, and one-input
provenance.

<a id="test-local-conceptual-example"></a>
### Test-local conceptual example

The following types are examples local to a test or explanation. They are not production [operation](../../glossary.md#operation) kinds or promises about a future operation-family [API](../../glossary.md#term-application-programming-interface-api).

```java
private enum SampleKind implements OperationKind {
    SAMPLE;

    private static final List<OperationSignature> SIGNATURES = List.of(
            OperationSignature.fixed(SampleAttrs.class, 1, 1));

    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}

private record SampleAttrs(int axis, boolean keepDimensions)
        implements OperationAttrs {}

OperationKind kind = SampleKind.SAMPLE;
OperationAttrs attrs = new SampleAttrs(2, true);
OperationAttrs emptyAttrs = NoOperationAttrs.INSTANCE;
Operation operation = new Operation(kind, attrs);
Operation equalOperation = new Operation(
        SampleKind.SAMPLE, new SampleAttrs(2, true));
```

The `SampleKind` declaration shows how an enum inherits `Enum.name()` and owns one stable [immutable](../../glossary.md#term-immutable)
[signature](../../glossary.md#term-signature) list. That signature accepts exactly `SampleAttrs`, one input, and one output. The
`SampleAttrs` record shows the intended structural value style: two records with the same [`axis`](../../glossary.md#term-axis)
and `keepDimensions` values compare equal. `NoOperationAttrs.INSTANCE` remains the [canonical](../../glossary.md#term-canonical-form) value
for a kind whose signature explicitly accepts no [parameters](../../glossary.md#term-parameter); it is not accepted by this [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters)
kind.

`new Operation(kind, attrs)` stores those exact non-null objects without copying or [normalization](../../glossary.md#term-normalization). The second construction uses equal component values, so `operation.equals(equalOperation)` is `true` and their [hash codes](../../glossary.md#hash-codes-and-content-digests) are equal. The accessors return the original objects: `operation.kind() == kind` and `operation.attrs() == attrs` are both `true`. At the end of the example, `kind.name()` is `"SAMPLE"`, `attrs` contains `axis = 2` and `keepDimensions = true`, and `emptyAttrs` is the [singleton](../../glossary.md#term-singleton) `NoOperationAttrs.INSTANCE` available for a parameterless sample kind.

The example demonstrates the implemented [descriptor](../../glossary.md#term-descriptor)'s construction, [ownership](../../glossary.md#term-ownership-meanings), exact attributes
compatibility, and record value semantics for an attributes-bearing test value. It does not create
a [graph](../../glossary.md#graph) [node](../../glossary.md#node), infer a result [shape](../../glossary.md#shape) or [data type](../../glossary.md#data-type--datatype), report [backend](../../glossary.md#backend) support,
select a [kernel](../../glossary.md#kernel), or execute computation. Production `BinaryArithmeticKind`,
`BinaryComparisonKind`, `BooleanLogicalKind`, `WhereSelectionKind`, `UnaryElementwiseKind`, and
`FloatingClassificationKind` provide parameterless families, and
`ScalarElementwiseKind` provides a parameterized family with `ScalarValueAttrs` and
`ClampRangeAttrs`; `CastKind` provides the parameterized cast family with `CastAttrs`. Additional
kind families are not implied by this example. [Compiler](../../glossary.md#term-compiler) behavior and executable support for
existing kinds are implemented in their [owners](../../glossary.md#term-owner) under the closed [admission](../../glossary.md#term-admission) rules; neither is
provided by constructing an operation-kind or attributes value.
[Binary](../../glossary.md#term-binary-descriptor-and-metadata) arithmetic, binary comparison, [unary](../../glossary.md#term-unary) [elementwise](../../glossary.md#term-elementwise), floating-classification, and [scalar](../../glossary.md#term-scalar)
elementwise semantics have current public [Tensor](../../glossary.md#tensor) expression methods, as do [boolean](../../glossary.md#data-type--datatype) logical and
conditional-selection semantics. Cast semantics also have current public Tensor expression
construction.


---

[Previous topic](operation-windows.md) · [Next topic](compiled-graph.md)
