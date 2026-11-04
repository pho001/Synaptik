# Padding, tiling and tensor composition

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="pad-and-tile-expressions"></a>
### Pad and tile expressions

`Tensor.pad(long[] before, long[] after, ScalarValue constantValue)` constructs a fresh constant-
pad expression. The two arrays must contain exactly one non-negative width per input axis. Construction
checks both references and ranks, clones both arrays, and then validates and derives metadata from
the private copies; it never retains or mutates caller-owned arrays. At axis `i`, a static input
extent produces this checked result extent:

```text
inputExtent + before[i] + after[i]
```

For example, Shape `[2, 3]`, before widths `[1, 0]`, and after widths `[2, 4]` produce Shape
`[5, 7]`. Static zero extents and empty arrays for a scalar are valid. Every Dimension category
uses the same canonical symbolic arithmetic. Zero widths preserve the exact input Dimension
reference, while non-zero widths are retained as an exact formula.

The supplied padding value must exactly match the input data type and is retained by exact
reference without conversion. All seven current data types are accepted, including exact INT64
values above `2^53`, raw FLOAT16 and BFLOAT16 patterns, and canonical BOOL. The retained `double`
constructs an exact FLOAT64 value, so it succeeds only for a FLOAT64 receiver. Neither overload
decides whether a future backend can represent or execute the request.

`Tensor.tile(long... repeats)` constructs a fresh complete-pattern tiling expression. Its array
must contain exactly one strictly positive repeat count per input axis, is cloned before element
inspection, and is neither retained nor mutated. At axis `i`, a static input extent produces the
checked product `inputExtent * repeats[i]`. Shape `[2, 3]` with repeats `[2, 4]` therefore produces
Shape `[4, 12]`. Tiling repeats the complete logical pattern, not each scalar into one consecutive
run. A static zero extent and an empty scalar request are valid. A dynamic dimension is retained
by exact reference when its repeat is one; any other positive repeat is retained as an exact
canonical symbolic product.

<a id="symbolic-shape-examples"></a>
#### Symbolic Shape examples

<a id="goal-and-inputs-37"></a>
##### Goal and inputs

Show how pad and tile preserve exact Shape relationships before a concrete value for named extent
`N` is known. The input Shape is `[N]`.

```text
pad before [2], after [3]:  N -> N + 5
tile repeats [4]:           N -> 4 * N
```

<a id="result-and-interpretation-37"></a>
##### Result and interpretation

Padding applies the before width and then the after width through checked symbolic addition;
tiling applies the positive repeat through checked symbolic multiplication. The resulting Shapes
retain `[N + 5]` and `[4 * N]`. These formulas describe logical extents only: construction does
not bind `N`, calculate a concrete element count, derive physical layout, materialize values, or
execute either operation. Checked static values, symbolic coefficients, and symbolic offsets fail
with `ArithmeticException` before result identity allocation if they overflow `long`.

Both methods preserve the input's exact data type and gradient-eligibility value. Every successful
call returns a distinct unlabeled, storage-free Tensor with unresolved layout, even for an
identity-like request and even when the input has resolved canonical, offset, strided, or
broadcast geometry. Padding and tiling require future output materialization; unlike reshape,
expand, permutation, rank editing, or non-empty slice, they are not ordinary input-view geometry.
The results respectively record `PadKind.PAD` with one normalized `PadAttrs`, or `TileKind.TILE`
with one normalized `TileAttrs`, and exact provenance `[input]`.

<a id="complete-pad-and-tile-expression-example"></a>
#### Complete pad-and-tile expression example

<a id="goal-and-inputs-38"></a>
##### Goal and inputs

Construct both expressions from one resolved Tensor and inspect their checked Shapes, immutable
attributes, operation identity, and deliberately unresolved storage-free results.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class PadTileExpressionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true));

        long[] before = {1, 0};
        long[] after = {2, 4};
        long[] repeats = {2, 4};
        Tensor padded = input.pad(before, after, ScalarValue.float32(-1.0f));
        Tensor tiled = input.tile(repeats);
        before[0] = 99;
        repeats[0] = 99;

        PadAttrs padAttrs = (PadAttrs) padded.provenance().orElseThrow()
                .operation().attrs();
        TileAttrs tileAttrs = (TileAttrs) tiled.provenance().orElseThrow()
                .operation().attrs();

        System.out.println(padded.descriptor().shape());
        System.out.println(padAttrs.before());
        System.out.println(padAttrs.after());
        System.out.println(padAttrs.constantValue());
        System.out.println(tiled.descriptor().shape());
        System.out.println(tileAttrs.repeats());
        System.out.println(padded.provenance().orElseThrow().operation().kind());
        System.out.println(tiled.provenance().orElseThrow().operation().kind());
        System.out.println(padded.provenance().orElseThrow().inputs().getFirst() == input
                && tiled.provenance().orElseThrow().inputs().getFirst() == input);
        System.out.println(padded.descriptor().layout().isEmpty()
                && tiled.descriptor().layout().isEmpty());
        System.out.println(padded.hostStorage().isEmpty()
                && tiled.hostStorage().isEmpty());
    }
}
```

<a id="meaningful-lines-and-intermediate-results-21"></a>
##### Meaningful lines and intermediate results

- Checked addition derives padded Shape `[5, 7]`; checked multiplication derives tiled Shape
  `[4, 12]`.
- Mutating the input arrays after construction cannot change the normalized attributes, which
  remain `[1, 0]`, `[2, 4]`, and `[2, 4]`.
- The padding constant is an exact FLOAT32 `ScalarValue`, retained without a binary64 intermediate.
- Both expressions retain `FLOAT32` and true gradient eligibility, record their distinct semantic
  kinds, and preserve the exact input Tensor in one-input provenance.
- The resolved input geometry does not propagate. Both fresh results remain unresolved and have
  no host storage because these calls describe materializing operations without performing them.

<a id="result-and-interpretation-38"></a>
##### Result and interpretation

The program prints:

```text
Shape[5, 7]
[1, 0]
[2, 4]
ScalarValue[dataType=FLOAT32, bits=0xBF800000]
Shape[4, 12]
[2, 4]
PAD
TILE
true
true
true
```

This output demonstrates rank-coupled request ownership, checked Shape derivation, exact semantic
attributes, one-input provenance, and unresolved storage-free results. It does not demonstrate
padding or repeated values, padding-constant conversion, gradient rules, graph capture,
materialization, compiler canonicalization, backend or ONNX lowering, or execution.

<a id="failures-and-useful-variations-34"></a>
##### Failures and useful variations

- A null array or typed padding value fails with `NullPointerException` naming that parameter. A
  length different from input rank, a negative padding width, a mismatched constant type, or a
  non-positive repeat fails with
  `IllegalArgumentException` before result identity allocation.
- Static extent, symbolic coefficient, or symbolic offset overflow propagates as
  `ArithmeticException` before result identity allocation. The attributes themselves can
  represent `Long.MAX_VALUE`; the input-dependent Tensor expression is where checked result
  geometry is enforced.
- Zero-width padding and repeat-one tiling still produce distinct explicit expression identities
  with unresolved layouts. Scalar calls use empty arrays and are valid.
- `boolInput.pad(before, after, ScalarValue.bool(false))` is valid. Passing a FLOAT64 value to that
  receiver fails rather than applying truthiness conversion.
- Identifier exhaustion can occur only at final derived Tensor construction, after all local
  immutable metadata has been prepared.

<a id="tensor-composition-expressions"></a>
### Tensor composition expressions

The current API provides static `Tensor.concat(int, Tensor...)`, static
`Tensor.stack(int, Tensor...)`, and instance `tensor.unstack(int)`. Concat and stack snapshot
ordered non-empty inputs, validate their existing or insertion axis and operation-specific exact
type/Shape rules, and return fresh unresolved-layout results with exact ordered provenance.

Unstack is an ordered convenience over scalar selection. It first normalizes an existing axis,
requires a static selected extent no larger than `Integer.MAX_VALUE`, and validates the complete
result count before creating any output. It then returns the immutable list equivalent to:

```java
List.of(input.select(axis, 0), input.select(axis, 1), /* ... */)
```

For input Shape `[2, 3, 4]` and axis `1`, the three outputs correspond to `SELECT(1, 0)`,
`SELECT(1, 1)`, and `SELECT(1, 2)`, each with Shape `[2, 4]`. Each output is a fresh,
unlabeled, storage-free Tensor with its own one-output `TensorProducer`,
`TensorProvenance.outputIndex() == 0`, and exact input `[input]`. The outputs do not share a
producer. Each also uses scalar select's layout rule: resolved non-empty input geometry can produce
a selected-stride/advanced-offset logical view, while unresolved input geometry or an empty result
Shape remains unresolved.

A zero selected extent returns the canonical immutable empty list without creating an operation,
producer, Tensor, or identifier. If identifier exhaustion occurs during a non-empty request,
earlier identifiers remain consumed, the exception propagates, and no partial list is returned.

This provenance shape intentionally differs from a future genuine multi-output operation such as
top-K. Unstack is N independently reproducible SELECT occurrences, so it has N producers and every
output index is zero. A future top-K values/indices pair would be one semantically indivisible
occurrence with one shared producer and distinct output indices. This comparison preserves the
shared multi-output provenance contract; it does not specify a top-K API.

<a id="complete-tensor-composition-expression-example"></a>
#### Complete tensor-composition expression example

<a id="goal-and-inputs-39"></a>
##### Goal and inputs

Construct CONCAT and STACK expressions, then unstack a static axis as repeated SELECT.

```java
Tensor first = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(2, 1), Optional.empty(), false));
Tensor second = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(2, 2), Optional.empty(), true));

Tensor concatenated = Tensor.concat(-1, first, second);
Tensor stacked = Tensor.stack(0, first, first);
List<Tensor> outputs = concatenated.unstack(1);
```

<a id="result-and-interpretation-39"></a>
##### Result and interpretation

`concatenated` has Shape `[2, 3]`; `stacked` has Shape `[2, 2, 1]`; and `outputs`
contains three Shape-`[2]` tensors. Output `i` records `SelectKind.SELECT` with
`SelectAttrs(1, i)`, one independent producer, and provenance output index zero. The result
demonstrates model metadata construction, not concatenated or selected values, gradients,
materialization, compiler capture, backend lowering, or execution.

<a id="failures-and-useful-variations-35"></a>
##### Failures and useful variations

Concat and stack reject null or empty input containers and indexed null elements before descriptor
inspection. Unstack rejects an invalid axis, dynamic selected extent, or count above
`Integer.MAX_VALUE` before output allocation. A zero count returns an immutable empty list
without consuming an identifier.

---

[Previous topic](scatter.md) · [Next topic](windows.md)
