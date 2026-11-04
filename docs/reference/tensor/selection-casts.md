# Conditional selection and casts

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="conditional-selection-expression"></a>
### Conditional-selection expression

The static `Tensor.where(condition, ifTrue, ifFalse)` method builds one elementwise conditional
selection expression. It constructs metadata and does not read a condition value or choose a
branch:

```text
branchShape = broadcast(ifTrue.shape, ifFalse.shape)
resultShape = broadcast(condition.shape, branchShape)
```

The condition must have exactly `DataType.BOOL`. The true and false branches must each be
floating, including either FP8 type; the shared floating-promotion rule derives the result
data type from the branches only. The first local broadcast proves branch compatibility. The
second proves that the condition can address their common result shape. Equal dimensions and
static singleton expansion are accepted, including locally provable dynamic cases; incompatible
or locally unprovable symbolic pairs fail.

Every valid call returns a fresh Tensor with the promoted branch data type, final three-way
broadcast shape, unresolved layout, no label, and no host storage. Result `requiresGrad` is
`ifTrue.requiresGrad || ifFalse.requiresGrad`; the non-differentiable BOOL condition does not
contribute. Provenance contains `WhereSelectionKind.WHERE`, `NoOperationAttrs.INSTANCE`, and exact
ordered inputs `[condition, ifTrue, ifFalse]`. Repeated branch references remain repeated, and
repeated calls receive fresh identities.

Construction is eager only for descriptor, operation, provenance, and identity metadata. It does
not inspect input storage, evaluate either branch, define eager or lazy evaluation order, copy or
convert values, simplify equal branches, create gradient-routing rules, capture a graph, map ONNX,
or execute backend code. Conditional `where` is also distinct from scalar-index `select`, which
belongs to the later indexing family.

Current package-private compiler autograd routes cotangents only to selected floating branches
when both branches and the output have one exact equal type; the BOOL condition is
non-differentiable. Mixed-floating branch promotion remains unsupported because no cross-floating
cotangent-conversion policy is selected.

<a id="complete-conditional-selection-example"></a>
#### Complete conditional-selection example

<a id="goal-and-inputs-16"></a>
##### Goal and inputs

Build a selection expression from a BOOL condition of shape `[2, 1]`, a gradient-eligible
`BFLOAT16` true branch of shape `[1, 3]`, and a `FLOAT64` false branch of shape `[2, 1]`. All three
inputs are storage-free leaves, so the example observes expression metadata rather than selected
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

- `BFLOAT16` and `FLOAT64` branches promote to `FLOAT64`; the BOOL condition does not participate
  in promotion.
- Broadcasting the branch shapes `[1, 3]` and `[2, 1]` first produces `[2, 3]`. Broadcasting the
  condition shape `[2, 1]` with that branch shape then retains `[2, 3]`.
- `Tensor.where` records exact ordered condition, true-branch, and false-branch references. The
  operation kind is `WHERE`, and the attributes are the canonical parameterless singleton.
- Only the true branch requests gradients, so the result's eligibility flag is true. This is
  descriptor metadata, not a generated backward computation or routing rule.

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

The output proves local branch promotion, ordered two-stage broadcasting, descriptor derivation,
fresh identity, and exact three-input provenance. It does not prove which branch value is chosen,
gradient routing, graph capture, ONNX mapping, backend support, or execution.

<a id="failures-and-useful-variations-14"></a>
##### Failures and useful variations

- A null argument fails in condition, true-branch, then false-branch order with the parameter name
  as the `NullPointerException` message.
- A non-BOOL condition fails before branch promotion. An integral or BOOL branch fails through
  floating promotion; no implicit cast is inserted.
- Incompatible branches fail before the condition shape is combined. A condition incompatible
  with the common branch shape fails in the second broadcast.
- Scalar, zero-sized, rank-mismatched, singleton-expanded, and locally provable symbolic shapes
  are valid. Different dynamic symbols and a dynamic dimension paired with a non-singleton static
  size remain locally unprovable.
- Validation failures happen before result identity allocation. Exhausting the factory's tensor-ID
  space instead fails after the local descriptor, operation, and provenance values are built.

<a id="cast-expressions"></a>
### Cast expressions

`Tensor.cast(targetDataType)` records an explicit request to convert each logical input value to a
target data type. The method accepts every ordered pair of current `DataType` values,
including both FP8 types, with the Model-owned value meaning below. A concrete
backend still advertises and implements only the pairs it supports.

The row is the source type and the column is the target type. `BITS` preserves the exact represented
bits; `RNE` rounds directly to the target with round-to-nearest, ties-to-even; `WIDEN` preserves an
exact finite value and applies the NaN mapping below; `TRUNC-SAT` truncates toward zero and then
saturates; `SIGNEXT` sign-extends; `LOW32` retains the low 32 two's-complement bits; `01` maps false
and true to positive zero and positive one; and `TRUTH` is false only for integer zero or either
floating signed zero.

| Source \\ Target | `FLOAT64` | `FLOAT32` | `BFLOAT16` | `FLOAT16` | `INT64` | `INT32` | `BOOL` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `FLOAT64` | `BITS` | `RNE` | direct `RNE` | direct `RNE` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `FLOAT32` | `WIDEN` | `BITS` | `RNE` | `RNE` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `BFLOAT16` | `WIDEN` | exact `WIDEN` | `BITS` | `RNE` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `FLOAT16` | `WIDEN` | exact `WIDEN` | `RNE` | `BITS` | `TRUNC-SAT` | `TRUNC-SAT` | `TRUTH` |
| `INT64` | `RNE` | `RNE` | direct `RNE` | direct `RNE` | `BITS` | `LOW32` | `TRUTH` |
| `INT32` | `RNE` | `RNE` | direct `RNE` | direct `RNE` | `SIGNEXT` | `BITS` | `TRUTH` |
| `BOOL` | `01` | `01` | `01` | `01` | `01` | `01` | `BITS` |

For either FP8 source, conversion to FLOAT32/FLOAT64 is exact widening; to integral/BOOL it
uses the same `TRUNC-SAT`/`TRUTH` rules. Conversion to another low format rounds directly,
while same-type conversion is `BITS`. Every other numeric source converts directly to FP8
with `RNE`, and BOOL maps to exact positive zero/one. E4M3's no-infinity overflow rule below
is an exception to the usual signed-infinity target behavior, not saturation.

Same-type conversion preserves every raw bit, including floating sign, subnormal representation,
and every NaN sign, quiet/signaling bit, and payload. `CastValueConversions.convert` returns the
exact source `ScalarValue` reference for this scalar case. `Tensor.cast`, however, always creates a
fresh expression occurrence; Tensor construction does not call the scalar oracle or evaluate
storage.

Finite floating-to-floating and integer-to-floating conversions choose the target value nearest
to the exact represented source value, with an exact midpoint selecting the result whose least
significant significand bit is zero. The conversion rounds once, directly to the requested target
format. Overflow normally produces signed infinity, except E4M3: magnitude 464 rounds to
finite 448, larger magnitudes and infinity input become canonical NaN `0x7f`. E5M2 magnitude
at or above 61440 becomes signed infinity. Underflow is gradual through target subnormals;
an exact zero/least-subnormal tie selects signed zero. Floating zero keeps its sign.

Lossy cross-floating conversion maps every NaN to the target's positive canonical quiet pattern:

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

Integral width conversion is exact sign extension from `INT32` to `INT64`; `INT64` to `INT32`
retains the low 32 bits, equivalently reduction modulo `2^32`. Thus
`0x0000000100000001L` becomes `1`, `Long.MIN_VALUE` becomes `0`, and `Long.MAX_VALUE` becomes
`-1`. Floating-to-integral conversion interprets the represented source, truncates a finite value
toward zero, and clamps it to the target's inclusive signed range. NaN becomes zero, positive
infinity becomes the target maximum, and negative infinity becomes the target minimum. Examples
include `2147483646.9 -> 2147483646`, `2147483647.9 -> Integer.MAX_VALUE`, binary64 bits
`0x43E0000000000000 -> Long.MAX_VALUE`, and binary64 bits
`0xC3E0000000000000 -> Long.MIN_VALUE`.

BOOL false converts to numeric positive zero and BOOL true to numeric positive one. Numeric
truthiness is false only for integer zero and floating `+0` or `-0`; every nonzero finite value,
positive or negative subnormal, infinity, and NaN is true. `ScalarValue` supplies only canonical
Boolean values, so this contract does not assign meaning to non-canonical backend BOOL bytes.

Direct rounding matters for BFLOAT16. The exact binary64 value with bits
`0x3FF0100000400000` (`1 + 2^-8 + 2^-30`) converts directly to BFLOAT16 `0x3F81`. A forbidden
intermediate FLOAT32 conversion instead produces `0x3F808000`, which then rounds to BFLOAT16
`0x3F80`. Likewise, exact binary64 `0x3FF02FFFFFC00000` converts directly to `0x3F81`, while a
FLOAT32 intermediate `0x3F818000` would double-round to `0x3F82`. Integer-to-BFLOAT16 follows the
same direct rule: INT32 `1077936129` converts to `0x4E81`, not the FLOAT32-mediated `0x4E80`.

Cast changes the result data type but not its logical dimensions. The result descriptor therefore
retains the input descriptor's exact immutable `Shape` reference. Its layout is always unresolved,
including when the input has resolved geometry and when source and target types are equal. A cast
may change element width, and expression construction does not decide whether storage can be
reused, copied, or materialized. The result has no label and no host storage.

Result gradient eligibility is true exactly when the input already requests gradients and both
the source and target data types are floating. This flag is descriptor metadata only. It neither
creates a backward rule nor promises that a backend supports differentiation through the cast.
Current package-private Compiler autograd supports every floating-to-floating CAST: a same-type
cast reuses the incoming cotangent, while a cross-floating cast creates one ordinary CAST back to
the source floating type. That reverse cast uses this same forward conversion contract; there is
no gradient-specific rounding or straight-through rule. A cast with an integral or BOOL source or
target has no cotangent.

Every call returns a fresh Tensor with a new factory identity. A same-type call is still an
explicit `CAST` expression rather than an early return of the receiver. Repeated calls remain
distinct, and a chain retains only its immediately preceding Tensor at each provenance link.
Compiler optimization later owns any legal redundant-cast or cast-chain simplification.

Provenance contains one `Operation` with `CastKind.CAST`, one fresh
`CastAttrs(targetDataType)`, and the exact one-input list `[input]`. The source type is read from
the input descriptor and is not duplicated in `CastAttrs`. Construction does not read or mutate
input storage, convert a value, attach output storage, capture a graph, or execute work. The public
stateless `CastValueConversions.convert(ScalarValue, DataType)` method is the scalar reference for
the value policy; it is not a Tensor evaluator, storage adapter, backend capability query, or
element-loop implementation.

<a id="complete-cast-expression-example"></a>
#### Complete cast-expression example

<a id="goal-and-inputs-17"></a>
##### Goal and inputs

Build a same-type `FLOAT32` cast from a tensor with resolved dense layout, then cast that explicit
result to `INT64`. The input requests gradients. The example observes descriptor, identity,
ownership, and provenance boundaries; it does not observe converted element values.

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

- The input has a resolved dense layout, but `input.cast(DataType.FLOAT32)` creates a fresh
  same-type result whose layout is unresolved. The method records the request instead of returning
  `input` or claiming physical storage reuse.
- The same-type floating cast retains the true gradient request. Casting that result to `INT64`
  sets eligibility to false because the target is not floating.
- Both results retain the exact `shape` object. The first provenance link points to `input`; the
  second points to `sameType`, so the chain remains explicit and local.
- Each operation uses `CAST`; its `CastAttrs` stores only the exact target data type.

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

The output proves fresh same-type identity, exact shape-reference retention, unresolved result
layout, the gradient-eligibility boundary, absent result label/storage, typed target attributes,
and immediate-input provenance. The separate value contract specifies that a represented
`FLOAT32` value becomes `INT64` by truncation and saturation, but this metadata-only example does
not evaluate such a value, prove that a compiler removes the first cast, or establish backend
support.

<a id="failures-and-useful-variations-15"></a>
##### Failures and useful variations

- A null target fails with `NullPointerException` and message `targetDataType` before result
  identity allocation.
- Scalar, zero-sized, fully static, and dynamic shapes are accepted without shape reconstruction.
- Integral-to-floating, BOOL-to-floating, integral-to-integral, and BOOL-related casts are valid
  expressions but always have false gradient eligibility because the source or target is
  non-floating.
- Exhausting the factory's tensor-ID space fails after the local descriptor, attributes,
  operation, and provenance values are built.


---

[Previous topic](arithmetic.md) · [Next topic](elementwise.md)
