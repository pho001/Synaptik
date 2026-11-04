# Binary arithmetic, comparisons and Boolean logic

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

<a id="binary-arithmetic-expressions"></a>
### Binary arithmetic expressions

The seven current expression methods build model semantics; they do not calculate element values:

| Method | Ordered meaning of `left.method(right)` |
|---|---|
| `add` | left plus the right addend |
| `sub` | left minus the right subtrahend |
| `mul` | left multiplied by the right factor |
| `div` | left divided by the right denominator |
| `minimum` | pairwise minimum of the left and right operands |
| `maximum` | pairwise maximum of the left and right operands |
| `pow` | left base raised to the right exponent |

The receiver is always the ordered left input and the argument is always the ordered right input.
This order remains in provenance even for operations whose mathematical result is commonly
commutative.

Floating ADD retains its primitive signed-zero rule: `-0 + +0` produces `+0`. The permitted
arithmetic-site DAZ/FTZ latitude does not relax that result sign.

Pairwise `minimum` and `maximum` propagate NaN, order infinities normally, and choose negative
zero for minimum or positive zero for maximum when comparing opposite signed zeros, independent
of operand order. Unequal non-NaN values use ordinary numeric order; equal nonzero candidates
produce that numeric value without identifying one source operand. FLOAT64 and FLOAT32 use their
represented IEEE-754 values; BFLOAT16 uses the value represented by its exact 16-bit storage
after existing promotion. They promise no NaN payload or bitwise result. The reduction names
remain `min` and `max`.

ADD, SUB, MUL, MIN, and MAX accept either two floating operands or two signed-integral operands.
Floating promotion follows the rule above: homogeneous low pairs retain their type,
unlike low pairs promote to FLOAT32, FLOAT32 wins over every low type, and FLOAT64 wins over
all other floating types. Integral pairs use `INT32 < INT64`, with an INT32 operand conceptually
sign-extended into the promoted INT64 domain. DIV and POW remain floating-only. BOOL and mixed
floating/integral pairs are rejected; callers use an explicit `cast` when cross-category conversion is intended. Every accepted method applies the existing
right-aligned local broadcast rule and creates a fresh `TensorDescriptor`. The result layout is
unresolved even when every shape
dimension is static because expression construction has not chosen storage geometry. Result
`requiresGrad` is the logical OR of the input requests; integral descriptors necessarily have it
false. That flag records gradient eligibility only and does not install a gradient rule.

Integral ADD, SUB, and MUL mean fixed-width two's-complement modular arithmetic in the promoted
result type: INT32 wraps modulo 2^32 and INT64 wraps modulo 2^64, reinterpreted as signed. Integral
MIN and MAX use ordinary signed order. Construction records these requests but neither reads
values nor detects overflow.

The fresh result has a new factory identity, no label, and no host storage. Its provenance stores
one `Operation` with the matching `BinaryArithmeticKind` and `NoOperationAttrs.INSTANCE`, followed
by the exact ordered input references `[left, right]`. Repeated calls create distinct results, and
even identities such as adding a stored zero or multiplying by a stored one are not simplified at
this boundary. Compiler first-order AD supports all seven promoted floating binary kinds.
Selected contributions reverse broadcasting with `sumToShape`, then cast back to the selected
input type when needed. MIN/MAX split exact numeric ties by one half; integral roles do not
receive cotangents. [ElementwiseGradientRules](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java)
owns these ordinary-expression formulas. Engine compilation/publication and qualified backend
execution are implemented separately; none is performed by constructing this Tensor.

<a id="complete-binary-expression-example"></a>
#### Complete binary-expression example

<a id="goal-and-inputs-14"></a>
##### Goal and inputs

Build a division expression from a `BFLOAT16` left tensor of shape `[2, 1]` and a `FLOAT32` right
tensor of shape `[1, 3]`. The left descriptor does not request gradients; the right descriptor
does. Both inputs are storage-free leaves, so the example demonstrates expression metadata rather
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

- `BFLOAT16` and `FLOAT32` promote to `FLOAT32`.
- Right alignment compares shape axes as `1` with `3` and `2` with `1`; each singleton expands,
  producing shape `[2, 3]`.
- `left.div(right)` preserves the non-commutative roles: `left` is the numerator and `right` is
  the denominator. The operation kind is `DIV`, and the parameterless attributes are the canonical
  singleton.
- The input gradient requests `false` and `true` combine to `true`. This result is eligibility
  metadata rather than a generated backward computation.

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

The output proves local type promotion, broadcasting, descriptor derivation, fresh identity, and
ordered provenance. It does not prove a numerical quotient, division-by-zero behavior, a gradient
formula, graph capture, backend support, or execution because none of those behaviors is part of
the current expression-construction API.

<a id="failures-and-useful-variations-11"></a>
##### Failures and useful variations

- A null right operand fails with `NullPointerException` and message `right`.
- Integral ADD, SUB, MUL, MIN, and MAX are valid. Integral DIV or POW fails before broadcasting
  and result-ID allocation. BOOL and mixed floating/integral pairs fail through numeric promotion;
  no implicit cast is inserted.
- Incompatible static dimensions, different dynamic symbols, and a dynamic dimension paired with
  a non-singleton static size fail through local broadcasting. Equal dynamic symbols and singleton
  expansion remain valid.
- Validation failures happen before result identity allocation. Exhausting the factory's tensor-ID
  space instead fails after the local descriptor, operation, and provenance values are built.

<a id="binary-comparison-expressions"></a>
### Binary comparison expressions

The six current comparison methods build ordered elementwise relation semantics; they do not read
or compare element values:

| Method | Ordered meaning of `left.method(right)` |
|---|---|
| `greaterThan` | Left value is strictly greater than the right value. |
| `greaterOrEqual` | Left value is greater than or equal to the right value. |
| `lessThan` | Left value is strictly less than the right value. |
| `lessOrEqual` | Left value is less than or equal to the right value. |
| `equalTo` | Left and right values compare equal. |
| `notEqualTo` | Left and right values compare unequal. |

The receiver remains the ordered left input and the argument remains the ordered right input for
every method. Equality and inequality retain this order in provenance even though their relation
is symmetric.

Each method accepts every same-category ordered pair from the floating or signed-integral types.
The common comparison domain uses the unchanged floating hierarchy or `INT32 < INT64`; an INT32
operand is conceptually sign-extended when the integral domain promotes to INT64. BOOL and mixed
floating/integral pairs are rejected unless the caller first records an explicit cast. The
promoted type is not the output type. Right-aligned local broadcasting derives the result shape.
The result data
type is always `BOOL`, its layout is unresolved, and `requiresGrad` is always false even when one
or both inputs request gradients.

Every valid call returns a fresh Tensor with a new factory identity, no label, and no host storage.
Its provenance contains one `Operation` with the exact matching `BinaryComparisonKind` and
`NoOperationAttrs.INSTANCE`, followed by exact ordered input references `[left, right]`. Repeated,
self, and symmetric calls are not interned, reordered, or canonicalized. FLOAT64 and FLOAT32 use
their represented IEEE-754 values in the promoted domain. BFLOAT16 uses the value represented by
its exact 16-bit storage; widening adds no new conversion or rounding contract. If either operand
is NaN, all four ordered relations are false. Negative and positive zero compare equal, so neither
strict relation holds and both inclusive relations hold in either operand order. `EQUAL` is exact
numeric equality rather than raw-bit or tolerance equality: NaN is unequal to every value,
including itself, and opposite signed zeros are equal. `NOT_EQUAL` is its logical complement.
Finite values and infinities otherwise use ordinary numeric order. Integral relations use
ordinary signed order; EQUAL and NOT_EQUAL compare exact promoted signed values. Compiler capture,
training-graph treatment, derivative policy, and backend execution remain later responsibilities.

<a id="complete-comparison-expression-example"></a>
#### Complete comparison-expression example

<a id="goal-and-inputs-15"></a>
##### Goal and inputs

Build a less-than-or-equal expression from a `BFLOAT16` left tensor of shape `[2, 1]` and a
`FLOAT64` right tensor of shape `[1, 3]`. The left input requests gradients, but the logical result
is a non-differentiable `BOOL` tensor. Both inputs are storage-free leaves, so the example observes
expression metadata rather than numerical comparison results.

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

- `BFLOAT16` and `FLOAT64` establish `FLOAT64` as the common comparison domain, but the result
  remains `BOOL`.
- Right alignment compares shape axes as `1` with `3` and `2` with `1`; singleton expansion
  produces shape `[2, 3]`.
- `left.lessOrEqual(right)` preserves the ordered relation: `left` is compared to `right`, the
  operation kind is `LESS_OR_EQUAL`, and the attributes are the canonical parameterless singleton.
- The true input gradient request does not propagate. A comparison result has false gradient
  eligibility and this construction adds no gradient rule.

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

The output proves floating-domain validation, broadcasting, fixed `BOOL` descriptor facts, fresh
identity, and ordered provenance. It does not evaluate any elementwise truth value or the fixed
NaN, signed-zero, and exact numeric-equality semantics, nor does it prove gradient handling by a
Compiler, backend support, or execution.

<a id="failures-and-useful-variations-12"></a>
##### Failures and useful variations

- A null right operand fails with `NullPointerException` and message `right`.
- Every INT32/INT64 ordered pair is valid. BOOL and mixed floating/integral pairs fail numeric
  validation before broadcasting and result-ID allocation; no implicit cast is inserted.
- Incompatible static dimensions, different dynamic symbols, and a dynamic dimension paired with
  a non-singleton static size fail through local broadcasting. Equal dynamic symbols and singleton
  expansion remain valid.
- Validation failures happen before result identity allocation. Exhausting the factory's tensor-ID
  space instead fails after the local descriptor, operation, and provenance values are built.

<a id="boolean-logical-expressions"></a>
### Boolean logical expressions

The three current logical methods build elementwise truth semantics from exact `BOOL` tensors;
they do not read or interpret stored truth bytes:

| Method | Ordered meaning and shape rule |
|---|---|
| `logicalAnd(right)` | Conjunction of ordered receiver and argument values; right-aligned broadcasting derives the result shape. |
| `logicalOr(right)` | Disjunction of ordered receiver and argument values; right-aligned broadcasting derives the result shape. |
| `logicalNot()` | Negation of the receiver values; the result retains the exact input `Shape` reference without broadcasting. |

AND and OR require both operands to have exactly `DataType.BOOL`; NOT requires the receiver to
have exactly `DataType.BOOL`. No floating or integral value is accepted as truth, and construction
does not insert a cast. Binary broadcasting uses the same local proof rules as other binary
expressions: equal dimensions and singleton expansion are accepted, while incompatible or locally
unprovable symbolic pairs fail. NOT performs no shape algebra and does not copy a resolved input
layout.

Every valid call returns a fresh Tensor with a new factory identity, fixed `BOOL` data type,
unresolved layout, false gradient eligibility, no label, and no host storage. Its provenance
contains the exact `BooleanLogicalKind` with `NoOperationAttrs.INSTANCE`. AND and OR preserve exact
ordered inputs `[receiver, right]`, including self-use; NOT records exactly `[receiver]`. Repeated
calls, commutative operand reversals, and double negation remain distinct expressions.

Construction is eager only for descriptor, operation, provenance, and identity metadata. It does
not short-circuit, inspect input storage, compute truth values, simplify self-use or double
negation, reorder commutative inputs, propagate gradient eligibility, create a gradient rule,
capture a graph, or execute a backend operation.

<a id="failures-and-useful-variations-13"></a>
#### Failures and useful variations

- A null AND or OR argument fails with `NullPointerException` and message `right`.
- A non-BOOL receiver or argument fails with `IllegalArgumentException`; binary checks the left
  input before the right input, and unary reports the receiver as `input`.
- Incompatible static shapes and locally unprovable symbolic pairs fail through local
  broadcasting. Equal symbols and singleton-to-symbol expansion are valid.
- Comparison results already have `BOOL` descriptors and can feed all three logical methods
  directly. This chaining constructs metadata only and does not calculate either comparison or
  logical truth values.
- Validation failures occur before result identity allocation. Identifier exhaustion occurs only
  after the fixed descriptor, operation, and provenance values have been built.

The current class stores no graph-local `NodeId` or `ValueId`, gradient object or trainable role,
publication state, typed element access, device buffer, runtime residency, prepared state, or
backend support. Provenance does not add any of those roles; they remain separate planned
contracts in their owning layers.


---

[Previous topic](storage-association.md) · [Next topic](selection-casts.md)
