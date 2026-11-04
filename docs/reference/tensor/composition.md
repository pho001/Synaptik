# [Padding](../../glossary.md#padding), [tiling](../../glossary.md#tiling) and [tensor composition](../../glossary.md#tensor-composition)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## How does the indexing domain grow?

Application or layer-building code calls these methods with Tensor descriptions and configuration.
The method snapshots mutable request arrays or input containers, while keeping the exact Tensor
references needed to describe the result. It does not copy their stored values during construction.
For axis $a$, let $d_a\geq0$ be its input extent, $b_a,c_a\geq0$ its before/after padding,
and $r_a>0$ its integer repeat count. Padding gives $d'_a=d_a+b_a+c_a$;
tiling gives $d'_a=r_a d_a$. Thus an extent of $3$ padded by $0$ before and $4$ after
becomes $7$, while repeating it four times gives $12$ positions. These are checked integer
sizes, not floating-point formulas. Compare [window extraction](windows.md) for overlapping
samples and [axis insertion](axis-transforms.md#expand-dimensions-and-squeeze-expressions)
for adding a singleton without adding elements.

**[Reader model](../../glossary.md#term-reader-model).** Padding adds [constant](../../glossary.md#term-constant)-filled border positions; tiling repeats a complete
pattern. Neither is just a [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) rename. For example, tiling conceptual `[a, b]` twice
requests `[a, b, a, b]`, not `[a, a, b, b]`.
**Concatenation** joins inputs along an existing [axis](../../glossary.md#term-axis); **stacking** adds a new axis to hold
same-shaped inputs. Joining two `[2, 3]` [tensors](../../glossary.md#tensor) along axis zero therefore gives `[4, 3]` with
concat, but `[2, 2, 3]` with stack. **Unstack** removes one axis by returning separate [scalar](../../glossary.md#term-scalar)
selections. These examples describe intended values and [shape](../../glossary.md#shape) relationships, not [eager](../../glossary.md#term-eager-evaluation) execution.

<a id="pad-and-tile-expressions"></a>
### Pad and tile expressions

`Tensor.pad(long[] before, long[] after, ScalarValue constantValue)` constructs a fresh [constant](../../glossary.md#term-constant)-
pad expression. The two arrays must contain exactly one non-negative width per input [axis](../../glossary.md#term-axis). Construction
checks both references and ranks, clones both arrays, and then validates and derives [metadata](../../glossary.md#term-metadata) from
the private copies; it never retains or mutates caller-owned arrays. At axis `i`, a static input
[extent](../../glossary.md#term-extent) produces this checked result extent:

$$
d'_i=d_i+b_i+c_i.
$$

Here $i$ is the selected axis, $d_i$ its input extent, and $b_i,c_i$ the corresponding
entries in `before` and `after`; all are non-negative integers measured in positions.

For example, [Shape](../../glossary.md#shape) `[2, 3]`, before widths `[1, 0]`, and after widths `[2, 4]` produce Shape
`[5, 7]`. Static zero extents and empty arrays for a [scalar](../../glossary.md#term-scalar) are valid. Every [Dimension](../../glossary.md#dimension) category
uses the same [canonical](../../glossary.md#term-canonical-form) symbolic arithmetic. Zero widths preserve the exact input Dimension
reference, while non-zero widths are retained as an exact formula.

The supplied [padding](../../glossary.md#padding) value must exactly match the input [data type](../../glossary.md#data-type--datatype) and is retained by exact
reference without conversion. All seven current data types are accepted, including exact [INT64](../../glossary.md#data-type--datatype)
values above `2^53`, raw [FLOAT16](../../glossary.md#term-float16) and [BFLOAT16](../../glossary.md#term-bfloat16) patterns, and canonical [BOOL](../../glossary.md#data-type--datatype). The retained `double`
constructs an exact [FLOAT64](../../glossary.md#term-float64) value, so it succeeds only for a FLOAT64 receiver. Neither overload
decides whether a selected [backend](../../glossary.md#backend) can represent or execute the request.

`Tensor.tile(long... repeats)` constructs a fresh complete-pattern [tiling](../../glossary.md#tiling) expression. Its array
must contain exactly one strictly positive repeat count per input axis, is cloned before [element](../../glossary.md#term-element)
inspection, and is neither retained nor mutated. At axis `i`, a static input extent produces the
checked product `inputExtent * repeats[i]`. Shape `[2, 3]` with repeats `[2, 4]` therefore produces
Shape `[4, 12]`. Tiling repeats the complete logical pattern, not each scalar into one consecutive
run. A static zero extent and an empty scalar request are valid. A dynamic dimension is retained
by exact reference when its repeat is one; any other positive repeat is retained as an exact
canonical symbolic product.

<a id="symbolic-shape-examples"></a>
#### Symbolic [Shape](../../glossary.md#shape) examples

<a id="goal-and-inputs-37"></a>
##### Goal and inputs

Show how pad and tile preserve exact [Shape](../../glossary.md#shape) relationships before a concrete value for named [extent](../../glossary.md#term-extent)
`N` is known. The input Shape is `[N]`.

```text
pad before [2], after [3]:  N -> N + 5
tile repeats [4]:           N -> 4 * N
```

<a id="result-and-interpretation-37"></a>
##### Result and interpretation

[Padding](../../glossary.md#padding) applies the before width and then the after width through checked symbolic addition;
[tiling](../../glossary.md#tiling) applies the positive repeat through checked symbolic multiplication. The resulting [Shapes](../../glossary.md#shape)
retain `[N + 5]` and `[4 * N]`. These formulas describe logical extents only: construction does
not bind `N`, calculate a concrete [element](../../glossary.md#term-element) count, derive physical [layout](../../glossary.md#layout), materialize values, or
execute either [operation](../../glossary.md#operation). Checked static values, symbolic coefficients, and symbolic offsets fail
with `ArithmeticException` before result [identity](../../glossary.md#term-identity-meanings) allocation if they [overflow](../../glossary.md#term-overflow) `long`.

Both methods preserve the input's exact [data type](../../glossary.md#data-type--datatype) and [gradient](../../glossary.md#term-gradient)-eligibility value. Every successful
call returns a distinct unlabeled, storage-free [Tensor](../../glossary.md#tensor) with unresolved layout, even for an
identity-like request and even when the input has resolved [canonical](../../glossary.md#term-canonical-form), offset, strided, or
broadcast geometry. Padding and tiling require future output [materialization](../../glossary.md#materialization); unlike reshape,
expand, [permutation](../../glossary.md#permutation), [rank](../../glossary.md#term-rank) editing, or non-empty [slice](../../glossary.md#slice), they are not ordinary input-[view](../../glossary.md#term-view-meanings) geometry.
The results respectively record `PadKind.PAD` with one normalized `PadAttrs`, or `TileKind.TILE`
with one normalized `TileAttrs`, and exact [provenance](../../glossary.md#provenance) `[input]`.

<a id="complete-pad-and-tile-expression-example"></a>
#### Complete pad-and-tile expression example

<a id="goal-and-inputs-38"></a>
##### Goal and inputs

Construct both expressions from one resolved [Tensor](../../glossary.md#tensor) and inspect their checked [Shapes](../../glossary.md#shape), [immutable](../../glossary.md#term-immutable)
attributes, [operation](../../glossary.md#operation) [identity](../../glossary.md#term-identity-meanings), and deliberately unresolved storage-free results.

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
        // The methods copy widths/repeats, so later array edits cannot change the request.
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

- Checked addition derives padded [Shape](../../glossary.md#shape) `[5, 7]`; checked multiplication derives tiled Shape
  `[4, 12]`.
- Mutating the input arrays after construction cannot change the normalized attributes, which
  remain `[1, 0]`, `[2, 4]`, and `[2, 4]`.
- The [padding](../../glossary.md#padding) [constant](../../glossary.md#term-constant) is an exact [FLOAT32](../../glossary.md#term-float32) [`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue), retained without a [binary64](../../glossary.md#floating-point-precision-and-rounding) intermediate.
- Both expressions retain `FLOAT32` and true gradient eligibility, record their distinct semantic
  kinds, and preserve the exact input Tensor in one-input provenance.<span data-glossary-navigation="true"> ([gradient glossary](../../glossary.md#term-gradient); [Tensor glossary](../../glossary.md#tensor); [provenance glossary](../../glossary.md#provenance))</span>
- The resolved input geometry does not propagate. Both fresh results remain unresolved and have
  no host storage because these calls describe materializing operations without performing them.<span data-glossary-navigation="true"> ([host storage glossary](../../glossary.md#host-storage))</span>

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

This output demonstrates [rank](../../glossary.md#term-rank)-coupled request [ownership](../../glossary.md#term-ownership-meanings), checked [Shape](../../glossary.md#shape) derivation, exact semantic
attributes, one-input [provenance](../../glossary.md#provenance), and unresolved storage-free results. It does not demonstrate
[padding](../../glossary.md#padding) or repeated values, padding-[constant](../../glossary.md#term-constant) conversion, [gradient](../../glossary.md#term-gradient) rules, [graph capture](../../glossary.md#capture-graph-ports-and-reconstruction),
[materialization](../../glossary.md#materialization), [compiler](../../glossary.md#term-compiler) [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), [backend](../../glossary.md#backend) or [ONNX](../../glossary.md#term-onnx) [lowering](../../glossary.md#lowering), or execution.

<a id="failures-and-useful-variations-34"></a>
##### Failures and useful variations

- A null array or typed [padding](../../glossary.md#padding) value fails with `NullPointerException` naming that parameter. A
  length different from input rank, a negative padding width, a mismatched constant type, or a
  non-positive repeat fails with<span data-glossary-navigation="true"> ([parameter glossary](../../glossary.md#term-parameter); [rank glossary](../../glossary.md#term-rank); [constant glossary](../../glossary.md#term-constant))</span>
  `IllegalArgumentException` before result [identity](../../glossary.md#term-identity-meanings) allocation.
- Static [extent](../../glossary.md#term-extent), symbolic coefficient, or symbolic offset [overflow](../../glossary.md#term-overflow) propagates as
  `ArithmeticException` before result identity allocation. The attributes themselves can
  represent `Long.MAX_VALUE`; the input-dependent Tensor expression is where checked result
  geometry is enforced.<span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor))</span>
- Zero-width padding and repeat-one tiling still produce distinct explicit expression identities
  with unresolved layouts. Scalar calls use empty arrays and are valid.<span data-glossary-navigation="true"> ([tiling glossary](../../glossary.md#tiling); [layouts glossary](../../glossary.md#layout); [Scalar glossary](../../glossary.md#term-scalar))</span>
- `boolInput.pad(before, after, ScalarValue.bool(false))` is valid. Passing a FLOAT64 value to that
  receiver fails rather than applying truthiness conversion.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../glossary.md#term-float64))</span>
- Identifier exhaustion can occur only at final derived Tensor construction, after all local
  immutable metadata has been prepared.<span data-glossary-navigation="true"> ([immutable glossary](../../glossary.md#term-immutable); [metadata glossary](../../glossary.md#term-metadata))</span>

<a id="tensor-composition-expressions"></a>
### [Tensor composition](../../glossary.md#tensor-composition) expressions

The current [API](../../glossary.md#term-application-programming-interface-api) provides static `Tensor.concat(int, Tensor...)`, static
`Tensor.stack(int, Tensor...)`, and instance `tensor.unstack(int)`. Concat and stack [snapshot](../../glossary.md#term-snapshot)
ordered non-empty inputs, validate their existing or insertion [axis](../../glossary.md#term-axis) and [operation](../../glossary.md#operation)-specific exact
type/[Shape](../../glossary.md#shape) rules, and return fresh unresolved-[layout](../../glossary.md#layout) results with exact ordered [provenance](../../glossary.md#provenance).

Unstack is an ordered convenience over [scalar](../../glossary.md#term-scalar) selection. It first normalizes an existing axis,
requires a static selected [extent](../../glossary.md#term-extent) no larger than `Integer.MAX_VALUE`, and validates the complete
result count before creating any output. It then returns the [immutable](../../glossary.md#term-immutable) list equivalent to:

```java
// Each selection has its own producer; a list does not turn them into one multi-output operation.
List.of(input.select(axis, 0), input.select(axis, 1), /* ... */)
```

For input Shape `[2, 3, 4]` and axis `1`, the three outputs correspond to `SELECT(1, 0)`,
`SELECT(1, 1)`, and `SELECT(1, 2)`, each with Shape `[2, 4]`. Each output is a fresh,
unlabeled, storage-free [Tensor](../../glossary.md#tensor) with its own one-output [`TensorProducer`](../../glossary.md#producer--tensorproducer),
`TensorProvenance.outputIndex() == 0`, and exact input `[input]`. The outputs do not share a
[producer](../../glossary.md#term-producer). Each also uses [scalar select](../../glossary.md#scalar-select)'s layout rule: resolved non-empty input geometry can produce
a selected-[stride](../../glossary.md#term-stride)/advanced-offset logical [view](../../glossary.md#term-view-meanings), while unresolved input geometry or an empty result
Shape remains unresolved.

A zero selected extent returns the [canonical](../../glossary.md#term-canonical-form) immutable empty list without creating an operation,
producer, Tensor, or identifier. If identifier exhaustion occurs during a non-empty request,
earlier identifiers remain consumed, the exception propagates, and no partial list is returned.

This provenance shape intentionally differs from a future genuine multi-output operation such as
top-K. Unstack is N independently reproducible SELECT occurrences, so it has N producers and every
output index is zero. A future top-K values/indices pair would be one semantically indivisible
[occurrence](../../glossary.md#term-occurrence) with one shared producer and distinct output indices. This comparison preserves the
shared multi-output provenance contract; it does not specify a top-K API.

<a id="complete-tensor-composition-expression-example"></a>
#### Complete [tensor](../../glossary.md#tensor)-composition expression example

<a id="goal-and-inputs-39"></a>
##### Goal and inputs

Construct CONCAT and STACK expressions, then unstack a static [axis](../../glossary.md#term-axis) as repeated SELECT.

```java
Tensor first = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(2, 1), Optional.empty(), false));
Tensor second = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.of(2, 2), Optional.empty(), true));

// Concat joins an existing axis; stack adds a new axis even when an input reference repeats.
Tensor concatenated = Tensor.concat(-1, first, second);
Tensor stacked = Tensor.stack(0, first, first);
List<Tensor> outputs = concatenated.unstack(1);
```

<a id="result-and-interpretation-39"></a>
##### Result and interpretation

`concatenated` has [Shape](../../glossary.md#shape) `[2, 3]`; `stacked` has Shape `[2, 2, 1]`; and `outputs`
contains three Shape-`[2]` [tensors](../../glossary.md#tensor). Output `i` records `SelectKind.SELECT` with
`SelectAttrs(1, i)`, one independent [producer](../../glossary.md#term-producer), and [provenance](../../glossary.md#provenance) output index zero. The result
demonstrates [model](../../glossary.md#term-model) [metadata](../../glossary.md#term-metadata) construction, not concatenated or selected values, [gradients](../../glossary.md#term-gradient),
[materialization](../../glossary.md#materialization), [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings), [backend](../../glossary.md#backend) [lowering](../../glossary.md#lowering), or execution.

<a id="failures-and-useful-variations-35"></a>
##### Failures and useful variations

Concat and stack reject null or empty input containers and indexed null [elements](../../glossary.md#term-element) before [descriptor](../../glossary.md#term-descriptor)
inspection. Unstack rejects an invalid [axis](../../glossary.md#term-axis), dynamic selected [extent](../../glossary.md#term-extent), or count above
`Integer.MAX_VALUE` before output allocation. A zero count returns an [immutable](../../glossary.md#term-immutable) empty list
without consuming an identifier.

---

[Previous topic](scatter.md) · [Next topic](windows.md)
