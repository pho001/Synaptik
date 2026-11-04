# Conditional selection and casts

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**[Reader model](../../glossary.md#term-reader-model).** `where` requests a per-position choice: use the true-branch value when the
corresponding [Boolean](../../glossary.md#data-type--datatype) condition is true, otherwise the false-branch value. This is not a
Java `if` that conditionally constructs only one branch, nor selection by an integer [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix).
A **cast** requests conversion to another [data type](../../glossary.md#data-type--datatype); it is not merely reinterpreting the same
bytes under a new name. **[Widening](../../glossary.md#term-widening)** moves to a more expressive representation; **[narrowing](../../glossary.md#term-narrowing)**
can round, [overflow](../../glossary.md#term-overflow), or lose information under the specific conversion rules below.
[Shape](../../glossary.md#shape) [broadcasting](../../glossary.md#broadcasting) chooses corresponding positions; type promotion chooses the common branch
type. Those are distinct decisions.

<a id="conditional-selection-expression"></a>
### Conditional-selection expression

The static `Tensor.where(condition, ifTrue, ifFalse)` method builds one [elementwise](../../glossary.md#term-elementwise) conditional
selection expression. It constructs [metadata](../../glossary.md#term-metadata) and does not read a condition value or choose a
branch:

```text
branchShape = broadcast(ifTrue.shape, ifFalse.shape)
resultShape = broadcast(condition.shape, branchShape)
```

The condition must have exactly `DataType.BOOL`. The true and false branches must each be
floating, including either [FP8](../../glossary.md#term-fp8) type; the shared floating-promotion rule derives the result
[data type](../../glossary.md#data-type--datatype) from the branches only. The first local broadcast proves branch compatibility. The
second proves that the condition can address their common result [shape](../../glossary.md#shape). Equal [dimensions](../../glossary.md#dimension) and
static [singleton](../../glossary.md#term-singleton) expansion are accepted, including locally provable dynamic cases; incompatible
or locally unprovable symbolic pairs fail.

Every valid call returns a fresh [Tensor](../../glossary.md#tensor) with the promoted branch data type, final three-way
broadcast shape, unresolved [layout](../../glossary.md#layout), no label, and no [host storage](../../glossary.md#host-storage). Result `requiresGrad` is
`ifTrue.requiresGrad || ifFalse.requiresGrad`; the non-[differentiable](../../glossary.md#derivative-and-chain-rule) [BOOL](../../glossary.md#data-type--datatype) condition does not
contribute. [Provenance](../../glossary.md#provenance) contains `WhereSelectionKind.WHERE`, `NoOperationAttrs.INSTANCE`, and exact
ordered inputs `[condition, ifTrue, ifFalse]`. Repeated branch references remain repeated, and
repeated calls receive fresh [identities](../../glossary.md#term-identity-meanings).

Construction is [eager](../../glossary.md#term-eager-evaluation) only for [descriptor](../../glossary.md#term-descriptor), [operation](../../glossary.md#operation), provenance, and [identity](../../glossary.md#term-identity-meanings) metadata. It does
not inspect input storage, evaluate either branch, define eager or [lazy evaluation](../../glossary.md#term-lazy-evaluation) order, copy or
convert values, simplify equal branches, create [gradient](../../glossary.md#term-gradient)-routing rules, [capture](../../glossary.md#term-capture-meanings) a [graph](../../glossary.md#graph), map [ONNX](../../glossary.md#term-onnx),
or execute [backend](../../glossary.md#backend) code. Conditional `where` is also distinct from [scalar](../../glossary.md#term-scalar)-index `select`, which
belongs to the later indexing family.

[Compiler](../../glossary.md#term-compiler) [autograd](../../glossary.md#autograd) routes cotangents to selected floating branches, reverses their [broadcasting](../../glossary.md#broadcasting)
with ordinary `sumToShape`, and casts a completed contribution back to its selected branch type
when promotion changed that type. The BOOL condition is non-differentiable. This is the current
branch-only policy in [ElementwiseGradientRules](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java),
not a same-type-only restriction or a straight-through rule for discrete roles.

<a id="complete-conditional-selection-example"></a>
#### Complete conditional-selection example

<a id="goal-and-inputs-16"></a>
##### Goal and inputs

Build a selection expression from a [BOOL](../../glossary.md#data-type--datatype) condition of [shape](../../glossary.md#shape) `[2, 1]`, a [gradient](../../glossary.md#term-gradient)-eligible
[`BFLOAT16`](../../glossary.md#term-bfloat16) true branch of shape `[1, 3]`, and a [`FLOAT64`](../../glossary.md#term-float64) false branch of shape `[2, 1]`. All three
inputs are storage-free [leaves](../../glossary.md#term-leaf), so the example observes expression [metadata](../../glossary.md#term-metadata) rather than selected
values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class WhereExpressionExample {
    public static void main(String[] args) {
        Tensor condition = TensorFactory.create(new TensorDescriptor(
                DataType.BOOL, Shape.of(2, 1), Optional.empty(), false));
        Tensor ifTrue = TensorFactory.create(new TensorDescriptor(
                DataType.BFLOAT16, Shape.of(1, 3), Optional.empty(), true));
        Tensor ifFalse = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT64, Shape.of(2, 1), Optional.empty(), false));

        Tensor result = Tensor.where(condition, ifTrue, ifFalse);
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
                + (provenance.inputs().get(0) == condition
                && provenance.inputs().get(1) == ifTrue
                && provenance.inputs().get(2) == ifFalse));
        System.out.println("fresh="
                + (result != condition && result != ifTrue && result != ifFalse));
    }
}
```

<a id="meaningful-lines-and-intermediate-results-2"></a>
##### Meaningful lines and intermediate results

- [`BFLOAT16`](../../glossary.md#term-bfloat16) and [`FLOAT64`](../../glossary.md#term-float64) branches promote to `FLOAT64`; the BOOL condition does not participate
  in promotion.<span data-glossary-navigation="true"> ([BOOL glossary](../../glossary.md#data-type--datatype))</span>
- [Broadcasting](../../glossary.md#broadcasting) the branch [shapes](../../glossary.md#shape) `[1, 3]` and `[2, 1]` first produces `[2, 3]`. Broadcasting the
  condition shape `[2, 1]` with that branch shape then retains `[2, 3]`.
- `Tensor.where` records exact ordered condition, true-branch, and false-branch references. The
  operation kind is <span data-glossary-navigation="true"> ([operation glossary](../../glossary.md#operation))</span>`WHERE`, and the attributes are the [canonical](../../glossary.md#term-canonical-form) parameterless [singleton](../../glossary.md#term-singleton).
- Only the true branch requests gradients, so the result's eligibility flag is true. This is
  descriptor metadata, not a generated backward computation or routing rule.<span data-glossary-navigation="true"> ([gradients glossary](../../glossary.md#term-gradient); [descriptor glossary](../../glossary.md#term-descriptor); [metadata glossary](../../glossary.md#term-metadata))</span>

<a id="result-and-interpretation-16"></a>
##### Result and interpretation

The program prints:

```text
type=FLOAT64
shape=Shape[2, 3]
layoutUnresolved=true
requiresGrad=true
unlabeled=true
storageFree=true
kind=WHERE
parameterless=true
orderedInputs=true
fresh=true
```

The output proves local branch promotion, ordered two-stage [broadcasting](../../glossary.md#broadcasting), [descriptor](../../glossary.md#term-descriptor) derivation,
fresh [identity](../../glossary.md#term-identity-meanings), and exact three-input [provenance](../../glossary.md#provenance). It does not prove which branch value is chosen,
[gradient](../../glossary.md#term-gradient) routing, [graph capture](../../glossary.md#capture-graph-ports-and-reconstruction), [ONNX](../../glossary.md#term-onnx) mapping, [backend](../../glossary.md#backend) support, or execution.

<a id="failures-and-useful-variations-14"></a>
##### Failures and useful variations

- A null argument fails in condition, true-branch, then false-branch order with the parameter name
  as the <span data-glossary-navigation="true"> ([argument glossary](../../glossary.md#term-argument); [parameter glossary](../../glossary.md#term-parameter))</span>`NullPointerException` message.
- A non-BOOL condition fails before branch promotion. An integral or BOOL branch fails through
  floating promotion; no implicit cast is inserted.<span data-glossary-navigation="true"> ([BOOL glossary](../../glossary.md#data-type--datatype))</span>
- Incompatible branches fail before the condition shape is combined. A condition incompatible
  with the common branch shape fails in the second broadcast.<span data-glossary-navigation="true"> ([shape glossary](../../glossary.md#shape))</span>
- Scalar, zero-sized, rank-mismatched, singleton-expanded, and locally provable symbolic shapes
  are valid. Different dynamic symbols and a dynamic dimension paired with a non-singleton static
  size remain locally unprovable.<span data-glossary-navigation="true"> ([Scalar glossary](../../glossary.md#term-scalar); [rank glossary](../../glossary.md#term-rank); [singleton glossary](../../glossary.md#term-singleton); [dimension glossary](../../glossary.md#dimension))</span>
- Validation failures happen before result identity allocation. Exhausting the factory's tensor-ID
  space instead fails after the local descriptor, operation, and provenance values are built.<span data-glossary-navigation="true"> ([Validation glossary](../../glossary.md#term-validation); [identity glossary](../../glossary.md#term-identity-meanings); [tensor glossary](../../glossary.md#tensor); [descriptor glossary](../../glossary.md#term-descriptor); [operation glossary](../../glossary.md#operation); [provenance glossary](../../glossary.md#provenance))</span>

<a id="cast-expressions"></a>
### Cast expressions

For the conversion vocabulary, **truncate toward zero** removes the fractional part
(`-2.9` becomes `-2`), while **saturate** limits an out-of-range result to the nearest allowed
endpoint. Keeping the low bits instead wraps and can change the sign. **Direct [rounding](../../glossary.md#term-rounding)** rounds
once into the requested format; **double rounding** goes through another format first and can
choose a different final value. These are different conversion meanings, not interchangeable
implementation shortcuts. The [matrix](../../glossary.md#term-matrix-meanings) and examples below specify which applies to each pair.

`Tensor.cast(targetDataType)` records an explicit request to convert each logical input value to a
[target](../../glossary.md#term-target) [data type](../../glossary.md#data-type--datatype). The method accepts every ordered pair of current [`DataType`](../../glossary.md#data-type--datatype) values,
including both [FP8](../../glossary.md#term-fp8) types, with the [Model](../../glossary.md#term-model)-owned value meaning below. A [concrete
backend](../../glossary.md#concrete-backend) still advertises and implements only the pairs it supports.

The row is the source type and the column is the target type. `BITS` preserves the exact [represented
bits](../../glossary.md#term-represented-bits); `RNE` rounds directly to the target with [round-to-nearest, ties-to-even](../../glossary.md#floating-point-precision-and-rounding); `WIDEN` preserves an
exact [finite](../../glossary.md#term-finite) value and applies the [NaN](../../glossary.md#term-nan) mapping below; `TRUNC-SAT` truncates toward zero and then
saturates; `SIGNEXT` sign-extends; `LOW32` retains the low 32 [two's-complement](../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) bits; `01` maps false
and true to [positive zero](../../glossary.md#nan-infinity-and-signed-zero) and positive one; and `TRUTH` is false only for integer zero or either
floating [signed zero](../../glossary.md#term-signed-zero).

| Source \\ Target | [`FLOAT64`](../../glossary.md#term-float64) | [`FLOAT32`](../../glossary.md#term-float32) | [`BFLOAT16`](../../glossary.md#term-bfloat16) | [`FLOAT16`](../../glossary.md#term-float16) | [`INT64`](../../glossary.md#data-type--datatype) | [`INT32`](../../glossary.md#data-type--datatype) | [`BOOL`](../../glossary.md#data-type--datatype) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `FLOAT64` | `BITS` | `RNE` | direct `RNE` | direct `RNE` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `FLOAT32` | `WIDEN` | `BITS` | `RNE` | `RNE` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `BFLOAT16` | `WIDEN` | exact `WIDEN` | `BITS` | `RNE` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `FLOAT16` | `WIDEN` | exact `WIDEN` | `RNE` | `BITS` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `INT64` | `RNE` | `RNE` | direct `RNE` | direct `RNE` | `BITS` | `LOW32` | `TRUTH` |
| `INT32` | `RNE` | `RNE` | direct `RNE` | direct `RNE` | `SIGNEXT` | `BITS` | `TRUTH` |
| `BOOL` | `01` | `01` | `01` | `01` | `01` | `01` | `BITS` |

For either FP8 source, conversion to FLOAT32/FLOAT64 is exact [widening](../../glossary.md#term-widening); to integral/BOOL it
uses the same `TRUNC-SAT`/`TRUTH` rules. Conversion to another low format rounds directly,
while same-type conversion is `BITS`. Every other numeric source converts directly to FP8
with `RNE`, and BOOL maps to exact positive zero/one. [E4M3](../../glossary.md#term-e4m3)'s no-[infinity](../../glossary.md#term-infinity) [overflow](../../glossary.md#term-overflow) rule below
is an exception to the usual signed-infinity target behavior, not [saturation](../../glossary.md#term-saturation).

Same-type conversion preserves every raw bit, including floating sign, [subnormal](../../glossary.md#term-subnormal) representation,
and every NaN sign, quiet/signaling bit, and [payload](../../glossary.md#artifact-envelope-and-atomic-publication). `CastValueConversions.convert` returns the
exact source [`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue) reference for this [scalar](../../glossary.md#term-scalar) case. `Tensor.cast`, however, always creates a
fresh expression [occurrence](../../glossary.md#term-occurrence); [Tensor](../../glossary.md#tensor) construction does not call the scalar [oracle](../../glossary.md#term-test-oracle) or evaluate
storage.

Finite floating-to-floating and integer-to-floating conversions choose the target value nearest
to the exact represented source value, with an exact midpoint selecting the result whose least
significant [significand](../../glossary.md#term-significand) bit is zero. The conversion rounds once, directly to the requested target
format. Overflow normally produces signed infinity, except E4M3: magnitude 464 rounds to
finite 448, larger magnitudes and infinity input become [canonical NaN](../../glossary.md#term-canonical-nan) `0x7f`. [E5M2](../../glossary.md#term-e5m2) magnitude
at or above 61440 becomes signed infinity. [Underflow](../../glossary.md#term-underflow) is gradual through target subnormals;
an exact zero/least-subnormal tie selects signed zero. Floating zero keeps its sign.

Lossy cross-floating conversion maps every NaN to the target's positive [canonical](../../glossary.md#term-canonical-form) quiet pattern:

| Target | Canonical NaN bits |
|---|---:|
| `FLOAT64` | `0x7FF8000000000000` |
| `FLOAT32` | `0x7FC00000` |
| `BFLOAT16` | `0x7FC0` |
| `FLOAT16` | `0x7E00` |
| `FLOAT8_E4M3` | `0x7F` |
| `FLOAT8_E5M2` | `0x7E` |

No current lossy pair targets `FLOAT64`, but its canonical pattern is fixed for the family. The
lossless widening directions preserve NaN sign, quiet/signaling state, and complete source
fraction by left-aligning that fraction and filling new low bits with zero:

```text
BFLOAT16 -> FLOAT32: targetFraction = sourceFraction << 16
BFLOAT16 -> FLOAT64: targetFraction = sourceFraction << 45
FLOAT32  -> FLOAT64: targetFraction = sourceFraction << 29
```

For example, BFLOAT16 `0x7F81` becomes FLOAT32 `0x7F810000`, negative BFLOAT16
`0xFFC1` becomes FLOAT64 `0xFFF8200000000000`, and FLOAT32 `0x7FA12345` becomes
FLOAT64 `0x7FF42468A0000000`. Floating NaN converts to integral zero and to BOOL `true`.

Integral width conversion is exact [sign extension](../../glossary.md#term-sign-extension) from `INT32` to `INT64`; `INT64` to `INT32`
retains the low 32 bits, equivalently reduction modulo `2^32`. Thus
`0x0000000100000001L` becomes `1`, `Long.MIN_VALUE` becomes `0`, and `Long.MAX_VALUE` becomes
`-1`. Floating-to-integral conversion interprets the represented source, truncates a finite value
toward zero, and clamps it to the target's inclusive signed range. NaN becomes zero, [positive
infinity](../../glossary.md#nan-infinity-and-signed-zero) becomes the target maximum, and [negative infinity](../../glossary.md#nan-infinity-and-signed-zero) becomes the target minimum. Examples
include `2147483646.9 -> 2147483646`, `2147483647.9 -> Integer.MAX_VALUE`, [binary64](../../glossary.md#floating-point-precision-and-rounding) bits
`0x43E0000000000000 -> Long.MAX_VALUE`, and binary64 bits
`0xC3E0000000000000 -> Long.MIN_VALUE`.

BOOL false converts to numeric positive zero and BOOL true to numeric positive one. Numeric
truthiness is false only for integer zero and floating `+0` or `-0`; every nonzero finite value,
positive or negative subnormal, infinity, and NaN is true. `ScalarValue` supplies only canonical
[Boolean](../../glossary.md#data-type--datatype) values, so this contract does not assign meaning to non-canonical [backend](../../glossary.md#backend) BOOL bytes.

Direct rounding matters for BFLOAT16. The exact binary64 value with bits
`0x3FF0100000400000` (`1 + 2^-8 + 2^-30`) converts directly to BFLOAT16 `0x3F81`. A forbidden
intermediate FLOAT32 conversion instead produces `0x3F808000`, which then rounds to BFLOAT16
`0x3F80`. Likewise, exact binary64 `0x3FF02FFFFFC00000` converts directly to `0x3F81`, while a
FLOAT32 intermediate `0x3F818000` would double-round to `0x3F82`. Integer-to-BFLOAT16 follows the
same direct rule: INT32 `1077936129` converts to `0x4E81`, not the FLOAT32-mediated `0x4E80`.

Cast changes the result data type but not its logical [dimensions](../../glossary.md#dimension). The result [descriptor](../../glossary.md#term-descriptor) therefore
retains the input descriptor's exact [immutable](../../glossary.md#term-immutable) [`Shape`](../../glossary.md#shape) reference. Its [layout](../../glossary.md#layout) is always unresolved,
including when the input has resolved geometry and when source and target types are equal. A cast
may change [element](../../glossary.md#term-element) width, and expression construction does not decide whether storage can be
reused, copied, or materialized. The result has no label and no [host storage](../../glossary.md#host-storage).

Result [gradient](../../glossary.md#term-gradient) eligibility is true exactly when the input already requests gradients and both
the source and target data types are floating. This flag is descriptor [metadata](../../glossary.md#term-metadata) only. It neither
creates a backward rule nor promises that a backend supports differentiation through the cast.
Current package-private [Compiler](../../glossary.md#term-compiler) [autograd](../../glossary.md#autograd) supports every floating-to-floating CAST: a same-type
cast reuses the incoming [cotangent](../../glossary.md#cotangent), while a cross-floating cast creates one ordinary CAST back to
the source floating type. That reverse cast uses this same forward conversion contract; there is
no gradient-specific rounding or straight-through rule. A cast with an integral or BOOL source or
target has no cotangent.

Every call returns a fresh Tensor with a new factory [identity](../../glossary.md#term-identity-meanings). A same-type call is still an
explicit `CAST` expression rather than an early return of the receiver. Repeated calls remain
distinct, and a chain retains only its immediately preceding Tensor at each [provenance](../../glossary.md#provenance) link.
Compiler optimization later owns any legal redundant-cast or cast-chain simplification.

Provenance contains one [`Operation`](../../glossary.md#operation) with `CastKind.CAST`, one fresh
`CastAttrs(targetDataType)`, and the exact one-input list `[input]`. The source type is read from
the input descriptor and is not duplicated in `CastAttrs`. Construction does not read or mutate
input storage, convert a value, attach output storage, [capture](../../glossary.md#term-capture-meanings) a [graph](../../glossary.md#graph), or execute work. The public
stateless `CastValueConversions.convert(ScalarValue, DataType)` method is the scalar reference for
the value policy; it is not a Tensor evaluator, storage adapter, [backend capability](../../glossary.md#backend-capability) query, or
element-loop implementation.

<a id="complete-cast-expression-example"></a>
#### Complete cast-expression example

<a id="goal-and-inputs-17"></a>
##### Goal and inputs

Build a same-type [`FLOAT32`](../../glossary.md#term-float32) cast from a [tensor](../../glossary.md#tensor) with resolved dense [layout](../../glossary.md#layout), then cast that explicit
result to [`INT64`](../../glossary.md#data-type--datatype). The input requests [gradients](../../glossary.md#term-gradient). The example observes [descriptor](../../glossary.md#term-descriptor), [identity](../../glossary.md#term-identity-meanings),
[ownership](../../glossary.md#term-ownership-meanings), and [provenance](../../glossary.md#provenance) boundaries; it does not observe converted [element](../../glossary.md#term-element) values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class CastExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        LayoutDescriptor inputLayout = LayoutDescriptor.contiguous(shape);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(inputLayout), true));

        Tensor sameType = input.cast(DataType.FLOAT32);
        Tensor integral = sameType.cast(DataType.INT64);
        TensorProvenance firstOrigin = sameType.provenance().orElseThrow();
        TensorProvenance secondOrigin = integral.provenance().orElseThrow();
        CastAttrs firstAttrs = (CastAttrs) firstOrigin.operation().attrs();
        CastAttrs secondAttrs = (CastAttrs) secondOrigin.operation().attrs();

        System.out.println("sameTypeFresh=" + (sameType != input));
        System.out.println("sameShape=" + (integral.descriptor().shape() == shape));
        System.out.println("layoutUnresolved=" + integral.descriptor().layout().isEmpty());
        System.out.println("sameTypeRequiresGrad="
                + sameType.descriptor().requiresGrad());
        System.out.println("integralRequiresGrad="
                + integral.descriptor().requiresGrad());
        System.out.println("unlabeled=" + integral.label().isEmpty());
        System.out.println("storageFree=" + integral.hostStorage().isEmpty());
        System.out.println("firstKind=" + firstOrigin.operation().kind());
        System.out.println("firstTarget=" + firstAttrs.targetDataType());
        System.out.println("firstInput=" + (firstOrigin.inputs().getFirst() == input));
        System.out.println("secondTarget=" + secondAttrs.targetDataType());
        System.out.println("immediateInput="
                + (secondOrigin.inputs().getFirst() == sameType));
    }
}
```

<a id="meaningful-lines-and-intermediate-results-3"></a>
##### Meaningful lines and intermediate results

- The input has a resolved dense [layout](../../glossary.md#layout), but `input.cast(DataType.FLOAT32)` creates a fresh
  same-type result whose layout is unresolved. The method records the request instead of returning
  `input` or claiming physical storage reuse.
- The same-type floating cast retains the true [gradient](../../glossary.md#term-gradient) request. Casting that result to [`INT64`](../../glossary.md#data-type--datatype)
  sets eligibility to false because the target is not floating.
- Both results retain the exact [`shape`](../../glossary.md#shape) object. The first [provenance](../../glossary.md#provenance) link points to `input`; the
  second points to `sameType`, so the chain remains explicit and local.
- Each [operation](../../glossary.md#operation) uses `CAST`; its `CastAttrs` stores only the exact [target](../../glossary.md#term-target) [data type](../../glossary.md#data-type--datatype).

<a id="result-and-interpretation-17"></a>
##### Result and interpretation

The program prints:

```text
sameTypeFresh=true
sameShape=true
layoutUnresolved=true
sameTypeRequiresGrad=true
integralRequiresGrad=false
unlabeled=true
storageFree=true
firstKind=CAST
firstTarget=FLOAT32
firstInput=true
secondTarget=INT64
immediateInput=true
```

The output proves fresh same-type [identity](../../glossary.md#term-identity-meanings), exact [shape](../../glossary.md#shape)-reference retention, unresolved result
[layout](../../glossary.md#layout), the [gradient](../../glossary.md#term-gradient)-eligibility boundary, absent result label/storage, typed [target](../../glossary.md#term-target) attributes,
and immediate-input [provenance](../../glossary.md#provenance). The separate value contract specifies that a represented
[`FLOAT32`](../../glossary.md#term-float32) value becomes [`INT64`](../../glossary.md#data-type--datatype) by [truncation](../../glossary.md#term-truncation-toward-zero) and [saturation](../../glossary.md#term-saturation), but this [metadata](../../glossary.md#term-metadata)-only example does
not evaluate such a value, prove that a [compiler](../../glossary.md#term-compiler) removes the first cast, or establish [backend](../../glossary.md#backend)
support.

<a id="failures-and-useful-variations-15"></a>
##### Failures and useful variations

- A null [target](../../glossary.md#term-target) fails with `NullPointerException` and message `targetDataType` before result
  identity allocation.<span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings))</span>
- [Scalar](../../glossary.md#term-scalar), zero-sized, fully static, and dynamic [shapes](../../glossary.md#shape) are accepted without shape reconstruction.
- Integral-to-floating, BOOL-to-floating, integral-to-integral, and BOOL-related casts are valid
  expressions but always have false gradient eligibility because the source or target is
  non-floating.<span data-glossary-navigation="true"> ([BOOL glossary](../../glossary.md#data-type--datatype); [gradient glossary](../../glossary.md#term-gradient))</span>
- Exhausting the factory's tensor-ID space fails after the local descriptor, attributes,
  operation, and provenance values are built.<span data-glossary-navigation="true"> ([tensor glossary](../../glossary.md#tensor); [descriptor glossary](../../glossary.md#term-descriptor); [operation glossary](../../glossary.md#operation); [provenance glossary](../../glossary.md#provenance))</span>


---

[Previous topic](arithmetic.md) · [Next topic](elementwise.md)
