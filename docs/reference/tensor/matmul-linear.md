# Matrix multiplication and linear projection

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="matrix-multiplication-expressions"></a>
### Matrix-multiplication expressions

`Tensor.matmul(right)` constructs one storage-free expression for matrix multiplication, written
mathematically as `left @ right`. A **contraction dimension** is the shared extent whose pairwise
products are summed. For rank-two-or-greater operands, the last two axes are matrix axes: the
left Shape ends in `[M, K]`, the right Shape ends in `[K, N]`, and `K` is the contraction
dimension. Earlier axes are **batch dimensions** and broadcast right-aligned.

A rank-one left operand is treated as `[1, K]` for the operation and its inserted row axis is
removed from the result. A rank-one right operand is treated as `[K, 1]` and its inserted column
axis is removed. This gives the complete rank/Shape relationship:

| Left Shape | Right Shape | Result Shape | Interpretation |
|---|---|---|---|
| `[K]` | `[K]` | `[]` | vector dot product, returned as a scalar |
| `[M, K]` | `[K]` | `[M]` | matrix times vector |
| `[K]` | `[K, N]` | `[N]` | vector times matrix |
| `[M, K]` | `[K, N]` | `[M, N]` | matrix times matrix |
| `[2, M, K]` | `[1, K, N]` | `[2, M, N]` | batched multiplication with singleton expansion |
| `[K]` | `[B, K, N]` | `[B, N]` | vector broadcast over a right batch |
| `[B, M, K]` | `[K]` | `[B, M]` | right vector broadcast over a left batch |

Read each row by first broadcasting the leading batch prefixes, then appending `M` unless the
left operand is rank one and appending `N` unless the right operand is rank one. Every retained
row, column, equal batch, singleton-opposing, or unpaired batch Dimension is the exact input
reference.

<a id="concrete-matrix-example"></a>
#### Concrete matrix example

The input values and Shapes are:

```text
left  Shape [2, 3] = [[1, 2, 3],
                      [4, 5, 6]]

right Shape [3, 2] = [[1, 2],
                      [3, 4],
                      [5, 6]]
```

Each output position is one left-row/right-column dot product:

```text
result[0,0] = 1*1 + 2*3 + 3*5 = 22
result[0,1] = 1*2 + 2*4 + 3*6 = 28
result[1,0] = 4*1 + 5*3 + 6*5 = 49
result[1,1] = 4*2 + 5*4 + 6*6 = 64
```

Therefore the mathematical result has Shape `[2, 2]` and values:

```text
[[22, 28],
 [49, 64]]
```

This calculation explains the recorded operation meaning. Calling `matmul` does not read the
input values or produce these values eagerly; the current model creates metadata only.

<a id="dynamic-dimensions-and-local-failures"></a>
#### Dynamic dimensions and local failures

Both ranks must be at least one. Two unequal static contraction dimensions fail immediately.
When either contraction dimension is unresolved, equality is deferred because it does not affect
the exact result Shape and both input descriptors remain in provenance. Batch broadcasting is
more restrictive because construction must select an exact output Dimension:

- equal unresolved dimensions retain the exact left reference;
- a static singleton retains the exact opposing reference;
- an unpaired leading batch dimension retains its exact input reference;
- an unresolved dimension paired with static non-singleton `S` selects that exact static result
  Dimension and defers the obligation that the unresolved extent bind to `1` or `S`;
- unequal unresolved batch dimensions fail because no exact result Dimension is locally
  derivable; and
- unequal static non-singleton batch dimensions fail.

These rules are local to MATMUL and do not change general `ShapeBroadcast`. Null checks, numeric
promotion, rank checks, static contraction checks, and leading-to-trailing batch checks all occur
before result identity allocation. Current package-private compiler verification proves, rejects,
or retains every deferred equality or singleton-or-equal obligation; concrete binding must
discharge any retained obligation before execution.

<a id="data-type-numerical-meaning-and-result-metadata"></a>
#### Data type, numerical meaning, and result metadata

`DataTypePromotion.promoteNumeric` accepts same-category floating pairs
(including both FP8 types) and same-category signed-integral pairs
(`INT32`, `INT64`). Unlike low pairs promote to FLOAT32; otherwise the promoted type
is the result type. BOOL and floating/integral pairs fail, and construction inserts no cast.

For a FLOAT64 result, pairwise products accumulate in FLOAT64. All four low types and FLOAT32
results accumulate in FLOAT32, with one final conversion for a low result. Under
the current FLOAT32 contract, exact broadcast/contraction mapping and contributor membership stay fixed;
every pairwise product participates exactly once under any aggregate tree with per-step FLOAT32
rounding, DAZ/FTZ, and only corresponding multiply/add fusion. The existing qualifying nonempty
final exact-zero publication choice remains local to the complete MATMUL cell. Standalone product
and pre-publication arithmetic do not gain that publication rule, a one-term contraction gains no
positive-zero identity or new FMA, and an empty contraction remains positive zero. This adds no
final-output tolerance, term loss, reduced precision, reciprocal substitution, or class change.
Integral products and sums use the promoted signed width modulo `2^32` or `2^64`; an empty
integral contraction produces zero.

Every valid call returns a fresh, unlabeled Tensor with the promoted type, exact derived Shape,
unresolved layout, no storage, and gradient eligibility equal to the logical OR of the operands.
Its one-output provenance has index zero and retains `MatmulKind.MATMUL`,
`NoOperationAttrs.INSTANCE`, and exact ordered inputs `[left, right]`. This metadata records a
gradient request, not a gradient rule. It neither captures or compiles a graph nor promises
compiler validation, backend lowering, a kernel, storage allocation, or execution.

<a id="linear-projection-convenience"></a>
### Linear-projection convenience

`input.linear(weight)` and `input.linear(weight, bias)` expose the conventional parameter layout
used by a linear projection without introducing a LINEAR operation. Weight has Shape
`[outFeatures, inFeatures]`, while MATMUL needs its right operand as
`[inFeatures, outFeatures]`. The public convenience is therefore literal composition:

```text
input.linear(weight)       = input.matmul(weight.transpose())
input.linear(weight, bias) = input.matmul(weight.transpose()).add(bias)
```

These receiver conveniences remain on `Tensor` because the input naturally owns a single-output
last-axis contraction. They own no parameters: the NN `Linear` module is the separate stateful
owner of weight and optional bias, while `Tensor.linear` is its stateless Model operation
composition. This differs from recurrent scan's multi-input, multi-output domain protocol, which
uses the focused static `RecurrentScan` namespace.

Input rank must be at least one. Its final Dimension is `inFeatures`; every preceding input
Dimension is retained in order, and the exact weight axis-zero `outFeatures` Dimension becomes
the final result Dimension. Weight must have rank two. Optional bias is not general broadcasting:
it must have exact rank one, and its sole Dimension must be structurally equal to weight
out-features. A null bias is invalid; omit bias by selecting the one-argument overload.

| Input | Weight | Bias | Result |
|---|---|---|---|
| `[K]` | `[N, K]` | absent or `[N]` | `[N]` |
| `[B, K]` | `[N, K]` | absent or `[N]` | `[B, N]` |
| `[B, T, K]` | `[N, K]` | absent or `[N]` | `[B, T, N]` |

Floating pairs and signed-integral pairs inherit current MATMUL promotion. The inner MATMUL also
inherits its family/dtype result contract described above. In the biased form, the product
and bias are promoted again by ordinary ADD, and each visible operation applies only its own named
sites: no cross-node fusion or algebraic rewrite is introduced. A wider bias may widen only the
final result. The linear conveniences add no numerical freedom, final-output envelope, or inserted
cast. Static unequal contraction Dimensions fail locally; equality
involving an unresolved contraction extent may remain for later compiler validation or binding.
Bias equality never defers: named and expression Dimensions must be structurally equal, and
identity-based unknowns must be the same unknown.

Every caller-controlled null, promotion, rank, static contraction, and bias failure is validated
before construction, so no partial expression or Tensor ID is created. A successful no-bias call
creates two fresh wrappers and IDs in PERMUTE-then-MATMUL order and returns the MATMUL product
directly. A successful biased call creates three in PERMUTE-then-MATMUL-then-ADD order. Tensor-ID
exhaustion after an earlier intermediate does not roll back that consumed ID.

The no-bias result ends in MATMUL provenance. The biased result ends in ADD provenance with
ordered inputs `[product, bias]`; the product exposes MATMUL inputs `[input, transposedWeight]`,
and the transposed weight exposes PERMUTE input `[weight]` with axes `[1, 0]`. Every link has one
output at provenance index zero. PERMUTE may preserve a resolved logical view layout, while
MATMUL and ADD results are unresolved, unlabeled, and storage-free. The biased ADD Shape is
structurally equal to the product Shape and reuses its exact ordered Dimension object references,
but the outer Shape objects may differ. No-bias returns the product itself.

<a id="complete-producer-chain-example"></a>
#### Complete producer-chain example

The example constructs a biased FLOAT32 projection from input `[2, 3]`, conventional weight
`[4, 3]`, and bias `[4]`, then inspects the primitive producer chain.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.Optional;

public final class LinearProducerChainExample {
    public static void main(String[] args) {
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true));
        Tensor weight = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(4, 3), Optional.empty(), true));
        Tensor bias = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(4), Optional.empty(), true));

        Tensor output = input.linear(weight, bias);
        TensorProducer add = output.provenance().orElseThrow().producer();
        Tensor product = add.inputs().getFirst();
        TensorProducer matmul = product.provenance().orElseThrow().producer();
        Tensor transposedWeight = matmul.inputs().get(1);
        TensorProducer permute = transposedWeight.provenance().orElseThrow().producer();
        PermutationAttrs attrs = (PermutationAttrs) permute.operation().attrs();

        System.out.println(output.descriptor().shape());
        System.out.println(add.operation().kind());
        System.out.println(add.inputs().get(0) == product && add.inputs().get(1) == bias);
        System.out.println(matmul.operation().kind());
        System.out.println(matmul.inputs().get(0) == input
                && matmul.inputs().get(1) == transposedWeight);
        System.out.println(permute.operation().kind());
        System.out.println(permute.inputs().getFirst() == weight);
        System.out.println(attrs.axes());
        System.out.println(output.provenance().orElseThrow().outputIndex() + ","
                + product.provenance().orElseThrow().outputIndex() + ","
                + transposedWeight.provenance().orElseThrow().outputIndex());
        System.out.println(product.descriptor().shape().equals(output.descriptor().shape()));
        System.out.println(product.descriptor().shape() == output.descriptor().shape());
    }
}
```

It prints:

```text
Shape[2, 4]
ADD
true
MATMUL
true
PERMUTE
true
[1, 0]
0,0,0
true
false
```

The final two lines distinguish structural Shape equality from outer Shape identity. Corresponding
axes still reuse the exact input/product Dimension references. The example proves model metadata
and ordered producer connectivity only; it does not evaluate numbers, capture a compiled graph,
construct gradients, prove compiler recognition, or promise backend fusion or execution.


---

[Previous topic](windows.md) · [Next topic](attention.md)
