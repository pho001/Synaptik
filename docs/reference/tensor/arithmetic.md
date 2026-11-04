# [Binary](../../glossary.md#term-binary-descriptor-and-metadata) arithmetic, comparisons and [Boolean](../../glossary.md#data-type--datatype) logic

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Which values are combined?

Application code or a layer builder calls `left.div(right)` on the numerator Tensor.
The method records both exact Tensor references, derives a result description, and reads no
stored numbers. A leaf input has no producing expression; it may, as in this example, have
no storage at all. Retaining inputs is not a copy of their values.
For finite real inputs $x_{i,0}$ and $y_{0,j}\ne0$, broadcasting shapes `[2, 1]`
and `[1, 3]` describes $z_{i,j}=x_{i,0}/y_{0,j}$ for $0\leq i<2$ and $0\leq j<3$.
Each original position can participate in several quotients. For $x_{0,0}=6$ and
$y_{0,1}=3$, the ideal $z_{0,1}=2$; the metadata example below does not evaluate it.
Floating exceptional values and rounding still follow the operation's rules.

Compare [selection and explicit casts](selection-casts.md) when you need a Boolean choice
or a deliberate type conversion, rather than arithmetic promotion.

**[Reader model](../../glossary.md#term-reader-model).** **Binary** means two inputs, not a binary number format.
**[Elementwise](../../glossary.md#term-elementwise)** arithmetic combines corresponding logical positions, with compatible
[broadcasting](shapes.md#broadcasting)<span data-glossary-navigation="true"> ([broadcasting glossary](../../glossary.md#broadcasting))</span> when [shapes](../../glossary.md#shape) differ. For conceptual rows `[2, 4]`
and `[1, 3]`, subtraction requests `[1, 1]`; it does not combine the row into one total.
A comparison instead requests one true/false result per position, and Boolean logic combines
those truth values. **[Provenance](../../glossary.md#provenance)** records which [operation](../../glossary.md#operation) and ordered inputs describe a result;
it is not a record of already calculated values. Numerical exceptions below describe eventual
arithmetic, not work done while constructing these expressions.

When reading [gradient](../../glossary.md#term-gradient) notes, [automatic differentiation](../../glossary.md#autograd) (**[AD](../../glossary.md#autograd)**) constructs [derivatives](../../glossary.md#term-derivative) from
operation rules. A **[cotangent](../../glossary.md#cotangent)** is an incoming [sensitivity](../../glossary.md#term-sensitivity) propagated backward from a selected
result; reversing broadcast must add sensitivities from every repeated use back into the original
input shape. **First-order** means derivatives of the original calculation, not derivatives of
those derivatives. “[Commutative](../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks)” means exchanging operands leaves the mathematical result
unchanged, but the recorded input order can still matter to provenance.

<a id="binary-arithmetic-expressions"></a>
### [Binary](../../glossary.md#term-binary-descriptor-and-metadata) arithmetic expressions

The seven current expression methods build [model](../../glossary.md#term-model) semantics; they do not calculate [element](../../glossary.md#term-element) values:

| Method | Ordered meaning of `left.method(right)` |
|---|---|
| `add` | left plus the right addend |
| `sub` | left minus the right subtrahend |
| `mul` | left multiplied by the right factor |
| `div` | left divided by the right denominator |
| `minimum` | pairwise minimum of the left and right operands |
| `maximum` | pairwise maximum of the left and right operands |
| `pow` | left base raised to the right exponent |

The receiver is always the ordered left input and the [argument](../../glossary.md#term-argument) is always the ordered right input.
This order remains in [provenance](../../glossary.md#provenance) even for operations whose mathematical result is commonly
[commutative](../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks).

Floating ADD retains its primitive [signed-zero](../../glossary.md#nan-infinity-and-signed-zero) rule: `-0 + +0` produces `+0`. The permitted
arithmetic-site [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz) latitude does not relax that result sign.

Pairwise `minimum` and `maximum` propagate [NaN](../../glossary.md#term-nan), order [infinities](../../glossary.md#nan-infinity-and-signed-zero) normally, and choose [negative
zero](../../glossary.md#nan-infinity-and-signed-zero) for minimum or [positive zero](../../glossary.md#nan-infinity-and-signed-zero) for maximum when comparing opposite signed zeros, independent
of operand order. Unequal non-NaN values use ordinary numeric order; equal nonzero candidates
produce that numeric value without identifying one source operand. [FLOAT64](../../glossary.md#term-float64) and [FLOAT32](../../glossary.md#term-float32) use their
represented [IEEE](../../glossary.md#term-ieee)-754 values; [BFLOAT16](../../glossary.md#term-bfloat16) uses the value represented by its exact 16-bit storage
after existing promotion. They promise no [NaN payload](../../glossary.md#nan-infinity-and-signed-zero) or bitwise result. The reduction names
remain `min` and `max`.

ADD, SUB, MUL, MIN, and MAX accept either two floating operands or two signed-integral operands.
Floating promotion follows the rule above: [homogeneous](../../glossary.md#term-homogeneous) low pairs retain their type,
unlike low pairs promote to FLOAT32, FLOAT32 wins over every low type, and FLOAT64 wins over
all other floating types. Integral pairs use `INT32 < INT64`, with an [INT32](../../glossary.md#data-type--datatype) operand conceptually
sign-extended into the promoted [INT64](../../glossary.md#data-type--datatype) domain. DIV and POW remain floating-only. [BOOL](../../glossary.md#data-type--datatype) and mixed
floating/integral pairs are rejected; callers use an explicit `cast` when cross-category conversion is intended. Every accepted method applies the existing
right-aligned local broadcast rule and creates a fresh `TensorDescriptor`. The result [layout](../../glossary.md#layout) is
unresolved even when every [shape](../../glossary.md#shape)
[dimension](../../glossary.md#dimension) is static because expression construction has not chosen storage geometry. Result
`requiresGrad` is the logical OR of the input requests; integral [descriptors](../../glossary.md#term-descriptor) necessarily have it
false. That flag records [gradient](../../glossary.md#term-gradient) eligibility only and does not install a gradient rule.

Integral ADD, SUB, and MUL use fixed-width two's-complement modular arithmetic in the promoted
result type: INT32 wraps modulo $2^{32}$ and INT64 modulo $2^{64}$, then interprets the
result as signed. These moduli count bit patterns, not a floating approximation. Integral
MIN and MAX use ordinary signed order. Construction neither reads values nor detects overflow.

The fresh result has a new factory [identity](../../glossary.md#term-identity-meanings), no label, and no [host storage](../../glossary.md#host-storage). Its provenance stores
one [`Operation`](../../glossary.md#operation) with the matching `BinaryArithmeticKind` and `NoOperationAttrs.INSTANCE`, followed
by the exact ordered input references `[left, right]`. Repeated calls create distinct results, and
even identities such as adding a stored zero or multiplying by a stored one are not simplified at
this boundary. [Compiler](../../glossary.md#term-compiler) first-order [AD](../../glossary.md#autograd) supports all seven promoted floating binary kinds.
Selected contributions reverse [broadcasting](../../glossary.md#broadcasting) with `sumToShape`, then cast back to the selected
input type when needed. MIN/MAX split exact numeric ties by one half; integral roles do not
receive cotangents. [ElementwiseGradientRules](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java)
owns these ordinary-expression formulas. [Engine](../../glossary.md#term-engine) [compilation](../../glossary.md#compiler-engine-and-execution-stage-names)/[publication](../../glossary.md#term-publication) and qualified [backend](../../glossary.md#backend)
execution are implemented separately; none is performed by constructing this [Tensor](../../glossary.md#tensor).

<a id="complete-binary-expression-example"></a>
#### Complete [binary](../../glossary.md#term-binary-descriptor-and-metadata)-expression example

<a id="goal-and-inputs-14"></a>
##### Goal and inputs

Build a division expression from a [`BFLOAT16`](../../glossary.md#term-bfloat16) left [tensor](../../glossary.md#tensor) of [shape](../../glossary.md#shape) `[2, 1]` and a [`FLOAT32`](../../glossary.md#term-float32) right
tensor of shape `[1, 3]`. The left [descriptor](../../glossary.md#term-descriptor) does not request [gradients](../../glossary.md#term-gradient); the right descriptor
does. Both inputs are storage-free [leaves](../../glossary.md#term-leaf), so the example demonstrates expression [metadata](../../glossary.md#term-metadata) rather
than numerical division.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class BinaryExpressionExample {
    public static void main(String[] args) {
        Tensor left = TensorFactory.create(new TensorDescriptor(
                DataType.BFLOAT16, Shape.of(2, 1), Optional.empty(), false));
        Tensor right = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(1, 3), Optional.empty(), true));

        // Record numerator/denominator order; constructing DIV does not read either input.
        Tensor result = left.div(right);
        TensorProvenance provenance = result.provenance().orElseThrow();

        System.out.println("type=" + result.descriptor().dataType());
        System.out.println("shape=" + result.descriptor().shape());
        System.out.println("layoutUnresolved=" + result.descriptor().layout().isEmpty());
        System.out.println("requiresGrad=" + result.descriptor().requiresGrad());
        System.out.println("unlabeled=" + result.label().isEmpty());
        System.out.println("storageFree=" + result.hostStorage().isEmpty());
        System.out.println("kind=" + provenance.operation().kind());
        System.out.println("parameterless="
                + (provenance.operation().attrs() == NoOperationAttrs.INSTANCE));
        System.out.println("orderedInputs="
                + (provenance.inputs().get(0) == left
                && provenance.inputs().get(1) == right));
        System.out.println("fresh=" + (result != left && result != right));
    }
}
```

<a id="meaningful-lines-and-intermediate-results"></a>
##### Meaningful lines and intermediate results

- [`BFLOAT16`](../../glossary.md#term-bfloat16) and [`FLOAT32`](../../glossary.md#term-float32) promote to `FLOAT32`.
- Pairing shape axes from the right compares `1` with `3` and `2` with `1`; each singleton
  expands, producing shape `[2, 3]`. This is unrelated to byte-address alignment.
- `left.div(right)` preserves the non-[commutative](../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) roles: `left` is the numerator and `right` is
  the denominator. The operation kind is <span data-glossary-navigation="true"> ([operation glossary](../../glossary.md#operation))</span>`DIV`, and the parameterless attributes are the canonical
  singleton.<span data-glossary-navigation="true"> ([canonical glossary](../../glossary.md#term-canonical-form))</span>
- The input [gradient](../../glossary.md#term-gradient) requests `false` and `true` combine to `true`. This result is eligibility
  metadata rather than a generated backward computation.<span data-glossary-navigation="true"> ([metadata glossary](../../glossary.md#term-metadata))</span>

<a id="result-and-interpretation-14"></a>
##### Result and interpretation

The program prints:

```text
type=FLOAT32
shape=Shape[2, 3]
layoutUnresolved=true
requiresGrad=true
unlabeled=true
storageFree=true
kind=DIV
parameterless=true
orderedInputs=true
fresh=true
```

The output proves local type promotion, [broadcasting](../../glossary.md#broadcasting), [descriptor](../../glossary.md#term-descriptor) derivation, fresh [identity](../../glossary.md#term-identity-meanings), and
ordered [provenance](../../glossary.md#provenance). It does not prove a numerical quotient, division-by-zero behavior, a [gradient](../../glossary.md#term-gradient)
formula, [graph capture](../../glossary.md#capture-graph-ports-and-reconstruction), [backend](../../glossary.md#backend) support, or execution because none of those behaviors is part of
the current expression-construction [API](../../glossary.md#term-application-programming-interface-api).

<a id="failures-and-useful-variations-11"></a>
##### Failures and useful variations

- A null right operand fails with `NullPointerException` and message `right`.
- Integral ADD, SUB, MUL, MIN, and MAX are valid. Integral DIV or POW fails before broadcasting
  and result-ID allocation. BOOL and mixed floating/integral pairs fail through numeric promotion;
  no implicit cast is inserted.<span data-glossary-navigation="true"> ([broadcasting glossary](../../glossary.md#broadcasting); [BOOL glossary](../../glossary.md#data-type--datatype); [numeric promotion glossary](../../glossary.md#numeric-promotion))</span>
- Incompatible static dimensions, different dynamic symbols, and a dynamic dimension paired with
  a non-singleton static size fail through local broadcasting. Equal dynamic symbols and singleton
  expansion remain valid.<span data-glossary-navigation="true"> ([dimensions glossary](../../glossary.md#dimension); [singleton glossary](../../glossary.md#term-singleton))</span>
- Validation failures happen before result identity allocation. Exhausting the factory's tensor-ID
  space instead fails after the local descriptor, operation, and provenance values are built.<span data-glossary-navigation="true"> ([Validation glossary](../../glossary.md#term-validation); [identity glossary](../../glossary.md#term-identity-meanings); [tensor glossary](../../glossary.md#tensor); [descriptor glossary](../../glossary.md#term-descriptor); [operation glossary](../../glossary.md#operation); [provenance glossary](../../glossary.md#provenance))</span>

<a id="binary-comparison-expressions"></a>
### [Binary](../../glossary.md#term-binary-descriptor-and-metadata) comparison expressions

The six current comparison methods build ordered [elementwise](../../glossary.md#term-elementwise) relation semantics; they do not read
or compare [element](../../glossary.md#term-element) values:

| Method | Ordered meaning of `left.method(right)` |
|---|---|
| `greaterThan` | Left value is strictly greater than the right value. |
| `greaterOrEqual` | Left value is greater than or equal to the right value. |
| `lessThan` | Left value is strictly less than the right value. |
| `lessOrEqual` | Left value is less than or equal to the right value. |
| `equalTo` | Left and right values compare equal. |
| `notEqualTo` | Left and right values compare unequal. |

The receiver remains the ordered left input and the [argument](../../glossary.md#term-argument) remains the ordered right input for
every method. Equality and inequality retain this order in [provenance](../../glossary.md#provenance) even though their relation
is symmetric.

Each method accepts every same-category ordered pair from the floating or signed-integral types.
The common comparison domain uses the unchanged floating hierarchy or `INT32 < INT64`; an [INT32](../../glossary.md#data-type--datatype)
operand is conceptually sign-extended when the integral domain promotes to [INT64](../../glossary.md#data-type--datatype). [BOOL](../../glossary.md#data-type--datatype) and mixed
floating/integral pairs are rejected unless the caller first records an explicit cast. The
promoted type is not the output type. Right-aligned local [broadcasting](../../glossary.md#broadcasting) derives the result [shape](../../glossary.md#shape).
The result [data
type](../../glossary.md#data-type--datatype) is always `BOOL`, its [layout](../../glossary.md#layout) is unresolved, and `requiresGrad` is always false even when one
or both inputs request [gradients](../../glossary.md#term-gradient).

Every valid call returns a fresh [Tensor](../../glossary.md#tensor) with a new factory [identity](../../glossary.md#term-identity-meanings), no label, and no [host storage](../../glossary.md#host-storage).
Its provenance contains one [`Operation`](../../glossary.md#operation) with the exact matching `BinaryComparisonKind` and
`NoOperationAttrs.INSTANCE`, followed by exact ordered input references `[left, right]`. Repeated,
self, and symmetric calls are not interned, reordered, or canonicalized. [FLOAT64](../../glossary.md#term-float64) and [FLOAT32](../../glossary.md#term-float32) use
their represented [IEEE](../../glossary.md#term-ieee)-754 values in the promoted domain. [BFLOAT16](../../glossary.md#term-bfloat16) uses the value represented by
its exact 16-bit storage; [widening](../../glossary.md#term-widening) adds no new conversion or [rounding](../../glossary.md#term-rounding) contract. If either operand
is [NaN](../../glossary.md#term-nan), all four ordered relations are false. Negative and [positive zero](../../glossary.md#nan-infinity-and-signed-zero) compare equal, so neither
strict relation holds and both inclusive relations hold in either operand order. `EQUAL` is exact
numeric equality rather than raw-bit or [tolerance](../../glossary.md#term-tolerance) equality: NaN is unequal to every value,
including itself, and opposite signed zeros are equal. `NOT_EQUAL` is its logical complement.
[Finite](../../glossary.md#term-finite) values and [infinities](../../glossary.md#nan-infinity-and-signed-zero) otherwise use ordinary numeric order. Integral relations use
ordinary signed order; EQUAL and NOT_EQUAL compare exact promoted signed values. [Compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings),
training-[graph](../../glossary.md#graph) treatment, [derivative](../../glossary.md#term-derivative) policy, and [backend](../../glossary.md#backend) execution remain later responsibilities.

<a id="complete-comparison-expression-example"></a>
#### Complete comparison-expression example

<a id="goal-and-inputs-15"></a>
##### Goal and inputs

Build a less-than-or-equal expression from a [`BFLOAT16`](../../glossary.md#term-bfloat16) left [tensor](../../glossary.md#tensor) of [shape](../../glossary.md#shape) `[2, 1]` and a
[`FLOAT64`](../../glossary.md#term-float64) right tensor of shape `[1, 3]`. The left input requests [gradients](../../glossary.md#term-gradient), but the logical result
is a non-[differentiable](../../glossary.md#derivative-and-chain-rule) [`BOOL`](../../glossary.md#data-type--datatype) tensor. Both inputs are storage-free [leaves](../../glossary.md#term-leaf), so the example observes
expression [metadata](../../glossary.md#term-metadata) rather than numerical comparison results.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class ComparisonExpressionExample {
    public static void main(String[] args) {
        Tensor left = TensorFactory.create(new TensorDescriptor(
                DataType.BFLOAT16, Shape.of(2, 1), Optional.empty(), true));
        Tensor right = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT64, Shape.of(1, 3), Optional.empty(), false));

        // The promoted comparison domain is not the result type: the result remains BOOL.
        Tensor result = left.lessOrEqual(right);
        TensorProvenance provenance = result.provenance().orElseThrow();

        System.out.println("type=" + result.descriptor().dataType());
        System.out.println("shape=" + result.descriptor().shape());
        System.out.println("layoutUnresolved=" + result.descriptor().layout().isEmpty());
        System.out.println("requiresGrad=" + result.descriptor().requiresGrad());
        System.out.println("unlabeled=" + result.label().isEmpty());
        System.out.println("storageFree=" + result.hostStorage().isEmpty());
        System.out.println("kind=" + provenance.operation().kind());
        System.out.println("parameterless="
                + (provenance.operation().attrs() == NoOperationAttrs.INSTANCE));
        System.out.println("orderedInputs="
                + (provenance.inputs().get(0) == left
                && provenance.inputs().get(1) == right));
        System.out.println("fresh=" + (result != left && result != right));
    }
}
```

<a id="meaningful-lines-and-intermediate-results-1"></a>
##### Meaningful lines and intermediate results

- [`BFLOAT16`](../../glossary.md#term-bfloat16) and [`FLOAT64`](../../glossary.md#term-float64) establish `FLOAT64` as the common comparison domain, but the result
  remains [`BOOL`](../../glossary.md#data-type--datatype).
- Pairing axes from the right compares `1` with `3` and `2` with `1`; singleton expansion
  produces shape `[2, 3]`. This is a coordinate pairing, not a byte-address requirement.
- `left.lessOrEqual(right)` preserves the ordered relation: `left` is compared to `right`, the
  operation kind is <span data-glossary-navigation="true"> ([operation glossary](../../glossary.md#operation))</span>`LESS_OR_EQUAL`, and the attributes are the [canonical](../../glossary.md#term-canonical-form) parameterless singleton.
- The true input gradient request does not propagate. A comparison result has false gradient
  eligibility and this construction adds no gradient rule.<span data-glossary-navigation="true"> ([gradient glossary](../../glossary.md#term-gradient))</span>

<a id="result-and-interpretation-15"></a>
##### Result and interpretation

The program prints:

```text
type=BOOL
shape=Shape[2, 3]
layoutUnresolved=true
requiresGrad=false
unlabeled=true
storageFree=true
kind=LESS_OR_EQUAL
parameterless=true
orderedInputs=true
fresh=true
```

The output proves floating-domain [validation](../../glossary.md#term-validation), [broadcasting](../../glossary.md#broadcasting), fixed [`BOOL`](../../glossary.md#data-type--datatype) [descriptor](../../glossary.md#term-descriptor) facts, fresh
[identity](../../glossary.md#term-identity-meanings), and ordered [provenance](../../glossary.md#provenance). It does not evaluate any [elementwise](../../glossary.md#term-elementwise) truth value or the fixed
[NaN](../../glossary.md#term-nan), [signed-zero](../../glossary.md#nan-infinity-and-signed-zero), and exact numeric-equality semantics, nor does it prove [gradient](../../glossary.md#term-gradient) handling by a
[Compiler](../../glossary.md#term-compiler), [backend](../../glossary.md#backend) support, or execution.

<a id="failures-and-useful-variations-12"></a>
##### Failures and useful variations

- A null right operand fails with `NullPointerException` and message `right`.
- Every INT32/INT64 ordered pair is valid. BOOL and mixed floating/integral pairs fail numeric
  validation before broadcasting and result-ID allocation; no implicit cast is inserted.<span data-glossary-navigation="true"> ([INT32 glossary](../../glossary.md#data-type--datatype); [INT64 glossary](../../glossary.md#data-type--datatype); [BOOL glossary](../../glossary.md#data-type--datatype); [validation glossary](../../glossary.md#term-validation); [broadcasting glossary](../../glossary.md#broadcasting))</span>
- Incompatible static dimensions, different dynamic symbols, and a dynamic dimension paired with
  a non-singleton static size fail through local broadcasting. Equal dynamic symbols and singleton
  expansion remain valid.<span data-glossary-navigation="true"> ([dimensions glossary](../../glossary.md#dimension); [singleton glossary](../../glossary.md#term-singleton))</span>
- Validation failures happen before result identity allocation. Exhausting the factory's tensor-ID
  space instead fails after the local descriptor, operation, and provenance values are built.<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings); [tensor glossary](../../glossary.md#tensor); [descriptor glossary](../../glossary.md#term-descriptor); [operation glossary](../../glossary.md#operation); [provenance glossary](../../glossary.md#provenance))</span>

<a id="boolean-logical-expressions"></a>
### [Boolean](../../glossary.md#data-type--datatype) logical expressions

The three current logical methods build [elementwise](../../glossary.md#term-elementwise) truth semantics from exact [`BOOL`](../../glossary.md#data-type--datatype) [tensors](../../glossary.md#tensor);
they do not read or interpret stored truth bytes:

| Method | Ordered meaning and [shape](../../glossary.md#shape) rule |
|---|---|
| `logicalAnd(right)` | Conjunction of ordered receiver and [argument](../../glossary.md#term-argument) values; right-aligned [broadcasting](../../glossary.md#broadcasting) derives the result shape. |
| `logicalOr(right)` | Disjunction of ordered receiver and argument values; right-aligned broadcasting derives the result shape. |
| `logicalNot()` | Negation of the receiver values; the result retains the exact input `Shape` reference without broadcasting. |

AND and OR require both operands to have exactly `DataType.BOOL`; NOT requires the receiver to
have exactly `DataType.BOOL`. No floating or integral value is accepted as truth, and construction
does not insert a cast. [Binary](../../glossary.md#term-binary-descriptor-and-metadata) broadcasting uses the same local proof rules as other binary
expressions: equal [dimensions](../../glossary.md#dimension) and [singleton](../../glossary.md#term-singleton) expansion are accepted, while incompatible or locally
unprovable symbolic pairs fail. NOT performs no shape algebra and does not copy a resolved input
[layout](../../glossary.md#layout).

Every valid call returns a fresh Tensor with a new factory [identity](../../glossary.md#term-identity-meanings), fixed `BOOL` [data type](../../glossary.md#data-type--datatype),
unresolved layout, false [gradient](../../glossary.md#term-gradient) eligibility, no label, and no [host storage](../../glossary.md#host-storage). Its [provenance](../../glossary.md#provenance)
contains the exact `BooleanLogicalKind` with `NoOperationAttrs.INSTANCE`. AND and OR preserve exact
ordered inputs `[receiver, right]`, including self-use; NOT records exactly `[receiver]`. Repeated
calls, [commutative](../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) operand reversals, and double negation remain distinct expressions.

Construction is [eager](../../glossary.md#term-eager-evaluation) only for [descriptor](../../glossary.md#term-descriptor), [operation](../../glossary.md#operation), provenance, and identity [metadata](../../glossary.md#term-metadata). It does
not short-circuit, inspect input storage, compute truth values, simplify self-use or double
negation, reorder commutative inputs, propagate gradient eligibility, create a gradient rule,
[capture](../../glossary.md#term-capture-meanings) a [graph](../../glossary.md#graph), or execute a [backend](../../glossary.md#backend) operation.

<a id="failures-and-useful-variations-13"></a>
#### Failures and useful variations

- A null AND or OR [argument](../../glossary.md#term-argument) fails with `NullPointerException` and message `right`.
- A non-[BOOL](../../glossary.md#data-type--datatype) receiver or argument fails with `IllegalArgumentException`; binary checks the left
  input before the right input, and unary reports the receiver as <span data-glossary-navigation="true"> ([binary glossary](../../glossary.md#term-binary-descriptor-and-metadata); [unary glossary](../../glossary.md#term-unary))</span>`input`.
- Incompatible static shapes and locally unprovable symbolic pairs fail through local
  broadcasting. Equal symbols and singleton-to-symbol expansion are valid.<span data-glossary-navigation="true"> ([shapes glossary](../../glossary.md#shape); [broadcasting glossary](../../glossary.md#broadcasting); [singleton glossary](../../glossary.md#term-singleton); [symbol glossary](../../glossary.md#dimension))</span>
- Comparison results already have `BOOL` descriptors and can feed all three logical methods
  directly. This chaining constructs metadata only and does not calculate either comparison or
  logical truth values.<span data-glossary-navigation="true"> ([descriptors glossary](../../glossary.md#term-descriptor); [metadata glossary](../../glossary.md#term-metadata))</span>
- Validation failures occur before result identity allocation. Identifier exhaustion occurs only
  after the fixed descriptor, operation, and provenance values have been built.<span data-glossary-navigation="true"> ([Validation glossary](../../glossary.md#term-validation); [identity glossary](../../glossary.md#term-identity-meanings); [operation glossary](../../glossary.md#operation))</span>

The current class stores no [graph](../../glossary.md#graph)-local [`NodeId`](../../glossary.md#nodeid) or [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot), [gradient](../../glossary.md#term-gradient) object or trainable role,
[publication](../../glossary.md#term-publication) state, typed [element](../../glossary.md#term-element) access, [device](../../glossary.md#term-device) [buffer](../../glossary.md#term-buffer), [runtime](../../glossary.md#term-runtime) [residency](../../glossary.md#residency), prepared state, or
[backend](../../glossary.md#backend) support. [Provenance](../../glossary.md#provenance) does not add any of those roles; they remain separate planned
contracts in their owning layers.


---

[Previous topic](storage-association.md) · [Next topic](selection-casts.md)
