# Gather-ND and scatter expressions

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

<a id="gather-nd-expressions"></a>
### Gather-ND expressions

The two public Gather-ND methods consume ordered logical inputs `[data, indices]`. The receiver is
`data`; `indices` must use exact `INT32` or `INT64` elements and have rank at least one. The
one-argument method is exactly the zero-batch form of `gatherNd(indices, batchDimensions)`.

The explicit non-negative batch count `B` must be smaller than both input ranks. Data and indices
Dimensions on axes `[0, B)` must be structurally equal: equal static sizes and equal dynamic
symbols pass, while broadcasting and new symbolic constraints are not introduced. The final
indices Dimension must have a statically known positive extent `K`, the tuple depth, no greater
than `dataRank - B`. Each tuple indexes data axes `[B, B + K)`.

For data rank `R` and indices rank `Q`, expression construction creates the result Shape once as:

```text
indices.shape[0:Q-1] + data.shape[B+K:R]
```

The result retains the exact Dimension references from the indices prefix and untouched data
suffix. If both parts are empty, `Shape.ofDimensions` returns the canonical rank-zero scalar
Shape. Every valid request creates a fresh, unlabeled, storage-free Tensor with the data Tensor's
exact type and gradient eligibility, unresolved layout, `GatherNdKind.GATHER_ND`, one exact
`GatherNdAttrs(B)`, and provenance containing exact references in `[data, indices]` order.

<a id="complete-gather-nd-expression-example"></a>
#### Complete Gather-ND expression example

<a id="goal-and-inputs-34"></a>
##### Goal and inputs

Construct zero-batch, batched, scalar-result, and dynamic-Dimension Gather-ND expressions. The
index tensors are storage-free because metadata construction needs their descriptors, not their
values.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class GatherNdExpressionExample {
    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }

    public static void main(String[] args) {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor zeroBatchIndices = tensor(DataType.INT32, Shape.of(5, 2), false);
        Tensor batchedIndices = tensor(DataType.INT64, Shape.of(2, 5, 1), false);
        Tensor zeroBatch = data.gatherNd(zeroBatchIndices);
        Tensor batched = data.gatherNd(batchedIndices, 1);

        Tensor scalar = tensor(DataType.BOOL, Shape.of(2, 3), false)
                .gatherNd(tensor(DataType.INT32, Shape.of(2), false));

        DynamicDimension batch = new DynamicDimension("N");
        DynamicDimension queries = new DynamicDimension("M");
        StaticDimension suffix = new StaticDimension(4);
        Tensor dynamicData = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(batch, new StaticDimension(3), suffix), true);
        Tensor dynamicIndices = tensor(DataType.INT64,
                Shape.ofDimensions(new DynamicDimension("N"), queries,
                        new StaticDimension(1)), false);
        Tensor dynamic = dynamicData.gatherNd(dynamicIndices, 1);

        GatherNdAttrs attrs = (GatherNdAttrs) batched.provenance().orElseThrow()
                .operation().attrs();
        System.out.println(zeroBatch.descriptor().shape());
        System.out.println(batched.descriptor().shape());
        System.out.println(scalar.descriptor().shape() == Shape.scalar());
        System.out.println(dynamic.descriptor().shape());
        System.out.println(attrs);
        System.out.println(dynamic.descriptor().shape().dimensions().get(1) == queries);
        System.out.println(dynamic.descriptor().shape().dimensions().get(2) == suffix);
        System.out.println(batched.provenance().orElseThrow().inputs().get(0) == data);
        System.out.println(batched.provenance().orElseThrow().inputs().get(1)
                == batchedIndices);
        System.out.println(batched.descriptor().requiresGrad()
                && batched.descriptor().layout().isEmpty()
                && batched.label().isEmpty()
                && batched.hostStorage().isEmpty());
    }
}
```

<a id="meaningful-lines-and-intermediate-results-19"></a>
##### Meaningful lines and intermediate results

- Zero-batch indices `[5, 2]` have tuple depth `2`. They retain prefix `[5]`, index data axes `0`
  and `1`, and leave data suffix `[4]`, producing `[5, 4]`.
- Batched indices `[2, 5, 1]` share leading Dimension `2`, retain indices prefix `[2, 5]`, index
  data axis `1`, and leave suffix `[4]`, producing `[2, 5, 4]`.
- Indices Shape `[2]` over data Shape `[2, 3]` has no retained indices prefix or data suffix, so
  the result is the canonical scalar `Shape.scalar()`.
- The dynamic case accepts independently constructed but structurally equal `N` batch Dimensions.
  Its result `[N, M, 4]` retains the exact `M` reference from indices and exact suffix Dimension
  reference from data.

<a id="result-and-interpretation-34"></a>
##### Result and interpretation

The program prints:

```text
Shape[5, 4]
Shape[2, 5, 4]
true
Shape[N, M, 4]
GatherNdAttrs[batchDimensions=1]
true
true
true
true
true
```

The output demonstrates both overloads, all four Shape cases, exact retained metadata and
Dimension identities, normalized attributes, and ordered provenance. It does not demonstrate
index-value validity, gathered values, gradient behavior, compiler capture, materialization,
backend lowering, or execution.

<a id="failures-and-useful-variations-31"></a>
##### Failures and useful variations

- Validation checks null data and indices, index type, indices rank, non-negative batch count,
  indices-rank fit, data-rank fit, batch-prefix equality, and tuple depth in that order. Every
  local failure occurs before result identifier allocation.
- A dynamic final indices Dimension fails because tuple depth controls which data axes remain in
  the result. Static depth zero or a depth greater than `dataRank - B` also fails.
- Different dynamic batch symbols fail structural equality; no broadcast or equality constraint
  is recorded.
- Construction never reads, normalizes, clamps, or bounds-checks an index value. Negative and
  out-of-range coordinate policy remains outside this model metadata boundary.
- Checked result-rank overflow propagates as `ArithmeticException`; identifier exhaustion can
  occur only during final derived Tensor construction.

<a id="gather-compatible-scatter-add-expressions"></a>
### Gather-compatible Scatter Add expressions

`data.scatterAdd(indices, updates, axis)` constructs one functional fixed-add expression with
ordered logical inputs `[data, indices, updates]`. It requires exact `INT32` or `INT64` indices,
exact matching data/update types, and floating or signed-integral data. BOOL is rejected. The raw
axis is normalized once against the data Shape. If normalized axis `a` selects data Shape
`[d0, ..., da, ..., dR-1]`, the exact updates Shape is:

```text
[d0, ..., d(a-1)] + indices.shape + [d(a+1), ..., d(R-1)]
```

This is exactly the Shape produced by `data.gather(indices, axis)`, including every static,
dynamic, or expression Dimension reference. There is no broadcast, symbolic binding, or selected-
extent compatibility rule. Scalar indices insert no Dimensions; an indices Shape containing a
zero extent produces a zero-element updates domain and remains structurally valid.

For an updates coordinate `[before..., i..., after...]`, where `i...` spans the complete indices
coordinate, the target is `[before..., indices[i...], after...]`. The conceptual target value
starts with the corresponding data value and adds every addressed update. Duplicate indices
therefore accumulate. Signed-integral addition is fixed-width modular; floating addition may be
reassociated and has no bitwise-order guarantee.

Every success returns a fresh, unlabeled, storage-free Tensor with the exact data Shape/type,
data/update gradient-eligibility OR, unresolved layout, `AxisScatterKind.SCATTER_ADD`, normalized
`IndexAxisAttrs`, exact provenance `[data, indices, updates]`, and output index zero. Construction
does not read index or update values, check index bounds, add values, mutate inputs, construct a
gradient, capture a graph, lower to a backend, or execute work. Failed local validation consumes
no Tensor identity; identifier exhaustion remains possible only during final derived creation.

<a id="complete-scatter-add-metadata-example"></a>
#### Complete Scatter Add metadata example

The following program exercises static rank-changing, symbolic, scalar-index, and empty-index
Shapes. All tensors are storage-free descriptor metadata.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Optional;

public final class ScatterAddExpressionExample {
    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }

    public static void main(String[] args) {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor indices = tensor(DataType.INT64, Shape.of(5, 6), false);
        Tensor updates = tensor(DataType.FLOAT32, Shape.of(2, 5, 6, 4), false);
        Tensor fixed = data.scatterAdd(indices, updates, -2);

        DynamicDimension batch = new DynamicDimension("batch");
        DynamicDimension features = new DynamicDimension("features");
        DynamicDimension queries = new DynamicDimension("queries");
        Tensor symbolicData = tensor(DataType.INT32,
                Shape.ofDimensions(batch, new StaticDimension(3), features), false);
        Tensor symbolicIndices = tensor(
                DataType.INT32, Shape.ofDimensions(queries), false);
        Tensor symbolicUpdates = tensor(DataType.INT32,
                Shape.ofDimensions(batch, queries, features), false);

        Tensor scalarIndices = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor scalarUpdates = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor emptyIndices = tensor(DataType.INT32, Shape.of(0, 6), false);
        Tensor emptyUpdates = tensor(DataType.FLOAT32, Shape.of(2, 0, 6, 4), false);

        System.out.println(fixed.descriptor().shape());
        System.out.println(fixed.provenance().orElseThrow().operation());
        System.out.println(symbolicData.scatterAdd(
                symbolicIndices, symbolicUpdates, 1).descriptor().shape());
        System.out.println(data.scatterAdd(
                scalarIndices, scalarUpdates, 1).descriptor().shape());
        System.out.println(data.scatterAdd(
                emptyIndices, emptyUpdates, 1).descriptor().shape());
        System.out.println(fixed.descriptor().requiresGrad());
        System.out.println(fixed.provenance().orElseThrow().inputs().get(0) == data);
        System.out.println(fixed.provenance().orElseThrow().inputs().get(1) == indices);
        System.out.println(fixed.provenance().orElseThrow().inputs().get(2) == updates);
    }
}
```

The first result Shape is `[2, 3, 4]`, while its operation is
`SCATTER_ADD` with `IndexAxisAttrs[axis=1]`. The symbolic call retains exact
`[batch, 3, features]` result Shape after requiring updates `[batch, queries, features]`.
The scalar and empty calls both retain result `[2, 3, 4]`; their required updates Shapes are
`[2, 4]` and `[2, 0, 6, 4]`. The final four lines are `true`, confirming data-derived gradient
eligibility and exact ordered provenance.

Null indices and updates fail in that order. Index type then fails before update type, numeric
data type, axis normalization, and exact updates-Shape equality. A Shape mismatch reports the
complete expected and actual Shapes. Bounds are value-aware: negative and out-of-range stored
indices are not normalized, wrapped, or clamped by model construction and remain obligations for
later compilation/preparation or execution.

The Shape replacement follows the terminology of official
[ONNX Gather](https://onnx.ai/onnx/operators/onnx__Gather.html). It intentionally differs from the
same-rank aligned contracts of [ONNX Scatter Elements](https://onnx.ai/onnx/operators/onnx__ScatterElements.html)
and [PyTorch `scatter_add`](https://docs.pytorch.org/docs/stable/generated/torch.scatter_add.html).
[JAX `lax.scatter_add`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter_add.html)
exposes general dimension-number mappings; this focused API does not adopt that broader language.

<a id="scatter-elements-expressions"></a>
### Scatter Elements expressions

The two public `scatterElements` overloads construct a functional same-rank update from ordered
logical inputs `[data, indices, updates]`. The receiver is unchanged base `data`; `indices`
selects one target coordinate along the normalized axis, and `updates` supplies the corresponding
value. The short overload selects `ScatterReduction.NONE`; the complete overload retains the
caller's exact non-null reduction.

Indices must use exact `INT32` or `INT64`, and updates must use the data tensor's exact type.
Indices and updates must have equal rank and structurally equal Shapes, that rank must equal the
data rank, and every non-selected indices Dimension must equal the corresponding data Dimension.
The selected extent may differ. For data Shape `[2, 3, 4]`, axis `1`, and indices/updates Shape
`[2, 5, 4]`, the result has exact data Shape `[2, 3, 4]`. At update coordinate `[0, 4, 2]`,
the corresponding index conceptually selects target `[0, index, 2]`.

`NONE` permits every current data type and requires unique targets. `ADD`, `MUL`, `MAX`,
and `MIN` permit floating and integral data and reject `BOOL`. Duplicate detection needs index
values and remains outside this metadata-only boundary.

For one concrete result coordinate `c`, let `U(c)` be the logical multiset of update scalar values
whose index positions select `c`. Every updates position contributes exactly one member; duplicate
targets remain distinct members even when their values are equal. A non-replacement reduction
combines exactly one occurrence of `data[c]` with exactly one occurrence of every member of
`U(c)`. If `U(c)` is empty, the result is the exact unchanged representation of `data[c]`; no
identity operation or canonicalization is permitted. Target membership, duplicate multiplicity
and untouched representations are exact regardless of traversal or scheduling.

Data and updates have the same declared type; no promotion or saturation changes that contract.
Low arithmetic nevertheless uses the specified FLOAT32 working/accumulator domain, with one
final narrowing at the declared low result. Its represented-value rules are:

- When `U(c)` is non-empty, floating `MUL` includes the base and every addressed update exactly
  once in a permitted rounded multiplication tree. It is **not** a once-rounded exact abstract
  product: an overflow-first `maxFinite * 2 * +0` may produce NaN. A NaN factor or a zero/infinity
  pair requires NaN. Named arithmetic sites obey their dtype's DAZ/FTZ and rounding rules;
  no public or saved boundary may be crossed by fusion. Finite tolerances are test-only, not a
  license to drop factors, widen arbitrary intermediates or change guards and value classes.
  See the [current aggregate floor](../../architecture/contracts/foundational-modules.md#aggregate-floor).
- When `U(c)` is non-empty, floating `MIN` and `MAX` propagate NaN without a payload, sign,
  signaling, source, or bitwise promise. Otherwise they use ordinary numeric order including
  infinities. If both zero signs occur, `MIN` produces negative zero and `MAX` positive zero,
  independent of encounter order. Equal non-zero candidates identify the represented numeric
  result, not a source occurrence.
- Integral `MUL` is exact-width two's-complement modular multiplication, modulo `2^32` for INT32
  and `2^64` for INT64. Integral `MIN` and `MAX` use ordinary signed order.

`NONE`, `ADD`, and the separate fixed `SCATTER_ADD` primitive retain their existing meanings.
`NONE` still requires one unique update for an addressed target. Configurable `ADD` still includes
the base and every addressed update, including duplicates, using fixed-width modular integral
addition or reassociable floating addition without a bitwise-order guarantee. These forward
semantics select no derivative or subgradient policy.

Every valid call returns a fresh, unlabeled, storage-free Tensor with exact data Shape/type,
data/update gradient-eligibility OR, unresolved layout, `AxisScatterKind.SCATTER_ELEMENTS`,
`ScatterElementsAttrs(normalizedAxis, reduction)`, and exact ordered provenance. Construction
reads no values, checks no index bound or duplicate target, applies no write or reduction, mutates
no input, and defines no derivative, compiler, materialization, backend, or execution behavior.

<a id="complete-scatter-elements-expression-example"></a>
#### Complete Scatter Elements expression example

<a id="goal-and-inputs-35"></a>
##### Goal and inputs

Construct replacement and maximum-reduction expressions for `FLOAT32` data Shape `[2, 3, 4]`
using aligned indices and updates Shape `[2, 5, 4]`.

```java
Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
Tensor indices = tensor(DataType.INT32, Shape.of(2, 5, 4), false);
Tensor updates = tensor(DataType.FLOAT32, Shape.of(2, 5, 4), false);

Tensor replacement = data.scatterElements(indices, updates, -2);
Tensor maximum = data.scatterElements(indices, updates, 1, ScatterReduction.MAX);
```

<a id="result-and-interpretation-35"></a>
##### Result and interpretation

Both results have Shape `[2, 3, 4]`. The negative axis normalizes to `1`; replacement records
`ScatterElementsAttrs[axis=1, reduction=NONE]`, while maximum records `MAX`. This proves
structural expression construction and reduction retention, not indexed writes or execution.

The current Metal backend executes only the canonical positive-rank `FLOAT32`/`INT32`/`FLOAT32`
replacement form above. It requires unique complete targets, validates every
bound before uniqueness and any selector dispatch or output write, copies addressed update and
unaddressed base bits exactly, and leaves all inputs unchanged. `MAX` and every other arithmetic
scatter, INT64, views, dynamic Shapes, Scatter-ND, output aliasing, and complete Metal backward
execution remain unsupported.

<a id="failures-and-useful-variations-32"></a>
##### Failures and useful variations

Null data, indices, updates, and explicit reduction fail in that order. Index type fails before
data/update type equality, reduction eligibility, axis normalization, and Shape checks. Shape
validation checks indices rank, updates rank, every updates/indices Dimension, then every
non-selected indices/data Dimension, all in increasing-axis order. Local failures occur before
result identity allocation.
<a id="scatter-nd-expressions"></a>
### Scatter-ND expressions

The three public Scatter-ND overloads construct functional tuple-index updates from ordered
logical inputs `[data, indices, updates]`. `data` is the receiver and unchanged base value.
`indices` supplies coordinate tuples, and `updates` supplies one value or suffix slice for each
tuple position. A target is the result coordinate or suffix slice addressed by one tuple; a
duplicate target occurs when multiple tuples address that same target.

The overloads delegate to one validation path:

| Public method | Reduction | Shared batch count |
|---|---|---|
| `scatterNd(indices, updates)` | `NONE` | `0` |
| `scatterNd(indices, updates, reduction)` | Exact supplied reduction | `0` |
| `scatterNd(indices, updates, reduction, batchDimensions)` | Exact supplied reduction | Exact supplied non-negative count |

All paths require exact `INT32` or `INT64` indices and exact matching data/update types.
`ScatterReduction.NONE` accepts every current data type. `ADD`, `MUL`, `MAX`, and `MIN` accept
floating or integral data and reject `BOOL`; no input is converted or promoted.

For data rank `R`, indices rank `Q`, shared-batch count `B`, and tuple depth `K` from the final
indices Dimension, construction requires `0 <= B < Q`, `B < R`, structurally equal data and
indices Dimensions on axes `[0, B)`, and `1 <= K <= R - B`. Tuple depth must be statically known.
The exact updates Shape is:

```text
indices.shape[0:Q-1] + data.shape[B+K:R]
```

The result retains the exact data Shape and type, combines data/update gradient eligibility by
logical OR, and has unresolved layout, no label, and no storage. Provenance records
`ScatterNdKind.SCATTER_ND`, one exact `ScatterNdAttrs(B, reduction)`, and the exact ordered
references `[data, indices, updates]`. Every valid request is fresh, including repeated requests
with identical operands and attributes.

<a id="complete-scatter-nd-expression-example"></a>
#### Complete Scatter-ND expression example

<a id="goal-and-inputs-36"></a>
##### Goal and inputs

Construct zero-batch replacement, zero-batch addition, explicit batched maximum, and scalar-update
Scatter-ND expressions. The inputs are storage-free because construction needs only immutable
descriptor metadata; it does not read any index or update value.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class ScatterNdExpressionExample {
    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }

    public static void main(String[] args) {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false);
        Tensor zeroBatchIndices = tensor(DataType.INT32, Shape.of(5, 2), false);
        Tensor zeroBatchUpdates = tensor(DataType.FLOAT32, Shape.of(5, 4), true);
        Tensor replacement = data.scatterNd(zeroBatchIndices, zeroBatchUpdates);
        Tensor addition = data.scatterNd(
                zeroBatchIndices, zeroBatchUpdates, ScatterReduction.ADD);

        Tensor batchedIndices = tensor(DataType.INT64, Shape.of(2, 5, 1), false);
        Tensor batchedUpdates = tensor(DataType.FLOAT32, Shape.of(2, 5, 4), false);
        Tensor batched = data.scatterNd(
                batchedIndices, batchedUpdates, ScatterReduction.MAX, 1);

        Tensor boolData = tensor(DataType.BOOL, Shape.of(2, 3), false);
        Tensor scalarUpdates = tensor(DataType.BOOL, Shape.scalar(), false);
        Tensor scalar = boolData.scatterNd(
                tensor(DataType.INT32, Shape.of(2), false), scalarUpdates);

        TensorProvenance origin = replacement.provenance().orElseThrow();
        ScatterNdAttrs defaultAttrs = (ScatterNdAttrs) origin.operation().attrs();
        ScatterNdAttrs additionAttrs = (ScatterNdAttrs) addition
                .provenance().orElseThrow().operation().attrs();
        ScatterNdAttrs batchedAttrs = (ScatterNdAttrs) batched
                .provenance().orElseThrow().operation().attrs();

        System.out.println(zeroBatchUpdates.descriptor().shape());
        System.out.println(replacement.descriptor().shape());
        System.out.println(batchedUpdates.descriptor().shape());
        System.out.println(batched.descriptor().shape());
        System.out.println(scalarUpdates.descriptor().shape() == Shape.scalar());
        System.out.println(scalar.descriptor().shape());
        System.out.println(defaultAttrs);
        System.out.println(additionAttrs);
        System.out.println(batchedAttrs);
        System.out.println(origin.inputs().get(0) == data
                && origin.inputs().get(1) == zeroBatchIndices
                && origin.inputs().get(2) == zeroBatchUpdates);
        System.out.println(replacement.descriptor().dataType() == DataType.FLOAT32
                && replacement.descriptor().requiresGrad()
                && replacement.descriptor().layout().isEmpty()
                && replacement.label().isEmpty()
                && replacement.hostStorage().isEmpty());
        System.out.println(replacement != addition
                && !replacement.id().equals(addition.id()));
    }
}
```

<a id="meaningful-lines-and-intermediate-results-20"></a>
##### Meaningful lines and intermediate results

- Zero-batch indices `[5, 2]` have tuple depth `2`. Their prefix `[5]` combines with the untouched
  data suffix `[4]`, so updates must be `[5, 4]`; the result remains data-shaped `[2, 3, 4]`.
- Batched indices `[2, 5, 1]` share leading Dimension `2`, use tuple depth `1` to address data axis
  `1`, and retain suffix `[4]`. Updates are `[2, 5, 4]`, and `ScatterNdAttrs` retains batch count
  `1` and `MAX`.
- Indices `[2]` over BOOL data `[2, 3]` have tuple depth `2` and leave no prefix or suffix, so the
  required updates Shape is the canonical scalar `Shape.scalar()`. The result remains `[2, 3]`.
- The short overload records `NONE` and batch count zero. The reduction overload records `ADD`
  and batch count zero. The true update eligibility on `zeroBatchUpdates` makes the replacement
  result eligible; indices never contribute eligibility.

<a id="result-and-interpretation-36"></a>
##### Result and interpretation

The program prints:

```text
Shape[5, 4]
Shape[2, 3, 4]
Shape[2, 5, 4]
Shape[2, 3, 4]
true
Shape[2, 3]
ScatterNdAttrs[batchDimensions=0, reduction=NONE]
ScatterNdAttrs[batchDimensions=0, reduction=ADD]
ScatterNdAttrs[batchDimensions=1, reduction=MAX]
true
true
true
```

The output proves overload defaults, both Shape formulas, canonical scalar updates, exact
attributes, ordered provenance, data/update eligibility OR, unresolved storage-free result
metadata, and fresh identity. It does not prove that an index is in bounds, detect duplicate
targets, apply replacement or reduction, calculate values, construct gradients, capture a graph,
select a backend, or execute work.

<a id="failures-and-useful-variations-33"></a>
##### Failures and useful variations

- Validation checks null data, indices, updates, and reduction; exact index type; exact
  data/update type; reduction eligibility; indices rank; non-negative batch count; indices-rank
  fit; data-rank fit; batch-prefix equality; tuple depth; and updates Shape in that order. Every
  local failure occurs before result identity allocation.
- A dynamic final indices Dimension fails because tuple depth determines which data axes are
  indexed. Static depth zero or greater than `dataRank - batchDimensions` also fails.
- Structurally equal dynamic batch symbols pass. Different symbols fail; construction creates no
  broadcast or compiler-style equality constraint.
- `BOOL` accepts `NONE` but rejects every arithmetic reduction before Shape validation. Floating
  and integral data accept all five reductions.
- Construction never reads, normalizes, clamps, or bounds-checks an index value. It does not
  detect the duplicate targets that make `NONE` invalid. Identifier exhaustion can occur only at
  final derived Tensor construction.

For arithmetic reduction, each tuple position contributes its complete update suffix slice
scalar by scalar. At one concrete result coordinate `c`, the base `data[c]` participates exactly
once and every addressed update scalar participates exactly once. Duplicate tuples contribute
distinct copies of every suffix-slice scalar, including equal values. A coordinate receiving no
update retains the exact base representation. `MUL`, `MIN`, and `MAX` use the same
order-independent floating and integral represented-value rules specified under
[Scatter Elements expressions](#scatter-elements-expressions). `NONE`, configurable `ADD`, and
the separate fixed `SCATTER_ADD` primitive remain unchanged. Scatter-ND forward semantics select
no derivative or subgradient policy.


---

[Previous topic](gather.md) · [Next topic](composition.md)
