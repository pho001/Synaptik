# Scaled dot-product attention

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="scaled-dot-product-attention-expressions"></a>
### Scaled dot-product attention expressions

Scaled dot-product attention maps a query sequence to weighted value rows by comparing each query
with every key. The receiver is the query. For right-broadcastable batch or head prefixes, the
Shapes are:

```text
query  [..., L, E]
key    [..., S, E]  -> attention scores [..., L, S]
value  [..., S, Ev] -> output           [..., L, Ev]
```

An **attention score** is one scaled query/key dot product. Eligible scores in each query row are
softmax-normalized over the final key-sequence axis `S`; the resulting weights select a weighted
sum of value rows. An absent scale means `1 / sqrt(E)` after the positive embedding extent is
bound. A present `ScalarValue` scale must be finite, strictly positive, floating, and exactly the
promoted query/key/value type.

Two parallel four-overload families keep result selection and optional semantics explicit:

| Call family | Result | Explicit mask | Scale and causal mode |
|---|---|---|---|
| `query.scaledDotProductAttention(key, value)` | output Tensor only | none | default scale, non-causal |
| `query.scaledDotProductAttention(key, value, attrs)` | output Tensor only | none | from `attrs` |
| `query.scaledDotProductAttention(key, value, mask)` | output Tensor only | exact BOOL mask | default scale, non-causal |
| `query.scaledDotProductAttention(key, value, mask, attrs)` | output Tensor only | exact BOOL mask | from `attrs` |
| `query.scaledDotProductAttentionWithWeights(key, value)` | output and weights | none | default scale, non-causal |
| `query.scaledDotProductAttentionWithWeights(key, value, attrs)` | output and weights | none | from `attrs` |
| `query.scaledDotProductAttentionWithWeights(key, value, mask)` | output and weights | exact BOOL mask | default scale, non-causal |
| `query.scaledDotProductAttentionWithWeights(key, value, mask, attrs)` | output and weights | exact BOOL mask | from `attrs` |

The original `scaledDotProductAttention` methods remain exact one-output occurrences. They create
only output slot zero: there is no hidden weights descriptor, weights wrapper, or second Tensor
identifier. The `WithWeights` methods return
`ScaledDotProductAttentionResult(output, weights)` from one exact two-output occurrence. They do
not reconstruct weights with a second operation.

`ScaledDotProductAttentionAttrs(Optional<ScalarValue> scale, boolean causal)` is immutable
operation-specific semantic state. It is not a general options container. Causal mode uses a
top-left-aligned **causal mask**: zero-based key position `j` is eligible for query position `i`
exactly when `j <= i`, including when `L` and `S` differ. An explicit BOOL mask uses `true` for an
eligible position and `false` for an excluded one. It must right-broadcast exactly to score Shape
`[..., L, S]`; scalar masks are valid. Explicit and causal eligibility combine with logical AND.

<a id="output-slots-and-shared-occurrence"></a>
#### Output slots and shared occurrence

Both occurrence forms use the same sole `SCALED_DOT_PRODUCT_ATTENTION` kind and exact
`ScaledDotProductAttentionAttrs` instance. Its single structural signature accepts three or four
ordered inputs and one or two outputs. Producer descriptor count distinguishes the requested
occurrence form; the operation kind and mathematical meaning do not change.

| Slot | Public accessor | Shape | Data type | Gradient eligibility metadata |
|---:|---|---|---|---|
| 0 | one-output return or `result.output()` | `[..., L, Ev]` | promoted query/key/value type | query OR key OR value |
| 1 | `result.weights()` | `[..., L, S]` | promoted query/key/value type | query OR key |

Both two-output descriptors have unresolved layout, and both wrappers are fresh, unlabeled, and
storage-free. They retain one exact shared `TensorProducer`, `Operation`, attributes reference,
ordered input snapshot, and corresponding descriptor references. Their provenance output indices
are zero and one. The BOOL mask and value Tensor do not contribute to weights gradient
eligibility; this metadata is only an eligibility request and does not define a gradient rule.

`ScaledDotProductAttentionResult` is a shallowly immutable record that retains its two exact
Tensor references. Its constructor checks only that neither component is null. Directly
constructing the record does not validate descriptor or producer agreement; the four
`WithWeights` methods supply the documented same-occurrence relationship.

<a id="shape-type-and-deferred-checks"></a>
#### Shape, type, and deferred checks

Query, key, and value must be floating and rank two or greater. Their leading prefixes broadcast
right-aligned together. Construction retains exact selected input Dimension references. A static
non-singleton can select an exact batch result while an unresolved participant leaves a future
singleton-or-equal obligation; unequal unresolved candidates fail when no exact output Dimension
can be derived. These rules are attention-specific and do not change general `ShapeBroadcast`.

Static unequal query/key embedding extents or key/value sequence extents fail locally, as does a
static-zero query embedding extent. When those extents are unresolved, equality or positivity is
handled by current package-private compiler verification and may remain for concrete binding. A
mask with an unresolved Dimension
against a static non-singleton score Dimension is accepted with the corresponding deferred
singleton-or-equal obligation; combinations that cannot preserve the already selected exact score
Shape fail locally.

Floating promotion occurs in query/key/value order. FLOAT64 scores and output sums accumulate in
FLOAT64. All four low types and FLOAT32 use FLOAT32 accumulation, with one final conversion for a
low output. Reassociation and fused multiply-add remain subject to attention's exact guards and
literal eligible-score EXP/sum/division sites. The distinct first-class finite-softmax shift does
not replace them; this contract gives no bitwise or cross-backend identical-rounding promise.

<a id="eligibility-and-special-values"></a>
#### Eligibility and special values

Mask and causal exclusion happen before score or value arithmetic. NaN or infinity in an excluded
key/value position therefore cannot contaminate a row. For each query row:

- no eligible positions, including an all-masked row, produce positive-zero weights and
  positive-zero output components;
- an eligible NaN score makes eligible weights and non-empty output components NaN;
- eligible positive-infinity scores split unit weight equally, while other eligible positions get
  positive-zero weight;
- all eligible scores at negative infinity produce positive-zero weights and output; and
- otherwise finite scores use ideal exponential ratios whose eligible weights total one ideally;
  eligible negative infinities and every excluded position get positive zero.

Eligible value NaN still follows ordinary floating multiply/add even when its eligible weight is
zero; only eligibility exclusion guarantees that a value is ignored. An empty key sequence is an
all-masked-row case for every existing query row. Empty query, value-feature, and batch axes retain
their ordinary Shape meanings.

<a id="runnable-output-and-weights-metadata-example"></a>
#### Runnable output-and-weights metadata example

This Java 26 program constructs only Tensor metadata, retrieves `output()` and `weights()` from
one result, and verifies their shared occurrence. Its concrete query, key, and value Shapes are
`[2, 4, 8]`, `[1, 6, 8]`, and `[2, 6, 10]`. Only value requests gradients, which makes the output
eligible while leaving weights ineligible.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.ScaledDotProductAttentionResult;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.Optional;

public final class AttentionWeightsMetadataExample {
    public static void main(String[] args) {
        Tensor query = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 4, 8), Optional.empty(), false));
        Tensor key = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(1, 6, 8), Optional.empty(), false));
        Tensor value = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 6, 10), Optional.empty(), true));
        var attrs = new ScaledDotProductAttentionAttrs(Optional.empty(), false);

        ScaledDotProductAttentionResult result =
                query.scaledDotProductAttentionWithWeights(key, value, attrs);
        Tensor output = result.output();
        Tensor weights = result.weights();
        TensorProvenance outputOrigin = output.provenance().orElseThrow();
        TensorProvenance weightsOrigin = weights.provenance().orElseThrow();

        System.out.println(output.descriptor().shape());
        System.out.println(weights.descriptor().shape());
        System.out.println(output.descriptor().dataType() + ","
                + weights.descriptor().dataType());
        System.out.println(output.descriptor().requiresGrad() + ","
                + weights.descriptor().requiresGrad());
        System.out.println(outputOrigin.producer() == weightsOrigin.producer());
        System.out.println(outputOrigin.outputIndex() + "," + weightsOrigin.outputIndex());
        System.out.println(outputOrigin.operation().attrs() == attrs);
        System.out.println(outputOrigin.inputs().size());
    }
}
```

It prints:

```text
Shape[2, 4, 10]
Shape[2, 4, 6]
FLOAT32,FLOAT32
true,false
true
0,1
true
3
```

The batch prefixes broadcast to `[2]`, giving output Shape `[2, 4, 10]` and weights Shape
`[2, 4, 6]`. The final four lines show one exact producer, slot order zero then one, the exact
attributes reference, and ordered inputs `[query, key, value]`. This proves current descriptors
and provenance only. It does not calculate attention values, capture a compiled graph, choose an
algorithm or backend, allocate result storage, or execute attention. Package-private compiler
first-order autograd can now construct query, key, and value cotangents from output slot zero and
query and key cotangents from canonical weights slot one for this exact two-output occurrence.
The one-output overload fails closed because it has no canonical same-occurrence weights output;
a BOOL mask remains non-differentiable.


---

[Previous topic](matmul-linear.md) · [Next topic](convolution.md)
