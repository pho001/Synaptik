# [Matrix multiplication](../../glossary.md#matrix-multiplication--matmul) and [linear projection](../../glossary.md#linear-projection)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## The expression builder supplies both operands

Application code or an NN `Linear` module calls `left.matmul(right)` with Tensor descriptions.
The method retains those exact references and describes a result; it does not copy their values.
`Tensor.linear` similarly composes a transpose, MATMUL and optional ADD; it does not acquire
parameter ownership. For the geometry of that transpose see [axis transforms](axis-transforms.md);
for the distinct state owner see [NN linear-layer initialization](../nn/composition.md#current-nn-automatic-parameter-initialization-contract).

**[Reader model](../../glossary.md#term-reader-model).** A [vector](../../glossary.md#term-vector) is a [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) array; a [matrix](../../glossary.md#term-matrix-meanings) is a [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) array with rows and columns.
Matrix multiplication combines each left row with each right column: multiply corresponding
components, then sum them. That sum is a **[dot product](../../glossary.md#term-dot-product)**, and the shared component [axis](../../glossary.md#term-axis) is
**contracted**, so it does not appear in the result. Earlier **[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)** axes repeat independent
matrix calculations. This is not [elementwise](../../glossary.md#term-elementwise) multiplication, which never performs that sum.
A **linear projection** uses [weights](../../glossary.md#term-weight) to turn input [features](../../glossary.md#term-feature) into output features; optional [bias](../../glossary.md#term-bias)
adds a separate value per output feature. The worked row/column example below shows the
[contraction](../../glossary.md#contraction-dimension) numerically without implying construction-time [evaluation](../../glossary.md#term-evaluation).

<a id="matrix-multiplication-expressions"></a>
### [Matrix](../../glossary.md#term-matrix-meanings)-multiplication expressions

`Tensor.matmul(right)` constructs one storage-free expression for [matrix multiplication](../../glossary.md#matrix-multiplication--matmul), written
mathematically as `left @ right`. A **[contraction dimension](../../glossary.md#contraction-dimension)** is the shared [extent](../../glossary.md#term-extent) whose pairwise
products are summed. For [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix)-or-greater operands, the last two [axes](../../glossary.md#term-axis) are matrix axes: the
left [Shape](../../glossary.md#shape) ends in `[M, K]`, the right Shape ends in `[K, N]`, and `K` is the contraction
dimension. Earlier axes are **[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [dimensions](../../glossary.md#dimension)** and broadcast right-aligned.

A [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) left operand is treated as `[1, K]` for the [operation](../../glossary.md#operation) and its inserted row axis is
removed from the result. A rank-one right operand is treated as `[K, 1]` and its inserted column
axis is removed. This gives the complete [rank](../../glossary.md#term-rank)/Shape relationship:

| Left Shape | Right Shape | Result Shape | Interpretation |
|---|---|---|---|
| `[K]` | `[K]` | `[]` | [vector](../../glossary.md#term-vector) [dot product](../../glossary.md#term-dot-product), returned as a [scalar](../../glossary.md#term-scalar) |
| `[M, K]` | `[K]` | `[M]` | matrix times vector |
| `[K]` | `[K, N]` | `[N]` | vector times matrix |
| `[M, K]` | `[K, N]` | `[M, N]` | matrix times matrix |
| `[2, M, K]` | `[1, K, N]` | `[2, M, N]` | batched multiplication with [singleton](../../glossary.md#term-singleton) expansion |
| `[K]` | `[B, K, N]` | `[B, N]` | vector broadcast over a right batch |
| `[B, M, K]` | `[K]` | `[B, M]` | right vector broadcast over a left batch |

Read each row by first [broadcasting](../../glossary.md#broadcasting) the leading batch prefixes, then appending `M` unless the
left operand is rank one and appending `N` unless the right operand is rank one. Every retained
row, column, equal batch, singleton-opposing, or unpaired [batch Dimension](../../glossary.md#batch-dimension) is the exact input
reference.

<a id="concrete-matrix-example"></a>
#### Concrete [matrix](../../glossary.md#term-matrix-meanings) example

The input values and [Shapes](../../glossary.md#shape) are:

```text
left  Shape [2, 3] = [[1, 2, 3],
                      [4, 5, 6]]

right Shape [3, 2] = [[1, 2],
                      [3, 4],
                      [5, 6]]
```

Let $A$ be the left $M$-by-$K$ matrix and $B$ the right $K$-by-$N$ matrix.
For row $0\leq i<M$, output column $0\leq j<N$, and shared coordinate $0\leq k<K$,
the ideal finite-real calculation is

$$
C_{i,j}=\sum_{k=0}^{K-1}A_{i,k}B_{k,j}.
$$

Here $M=N=2$ and $K=3$. The four cells are

$$
\begin{aligned}
C_{0,0}&=1(1)+2(3)+3(5)=22,& C_{0,1}&=1(2)+2(4)+3(6)=28,\\
C_{1,0}&=4(1)+5(3)+6(5)=49,& C_{1,1}&=4(2)+5(4)+6(6)=64.
\end{aligned}
$$

Each cell summarizes three matching products; the shared coordinate disappears from the output.
This real-number formula does not choose a floating summation tree, authorize extra fused
operations, or override the empty-contraction, signed-zero and result-rounding rules below.

Therefore the mathematical result has Shape `[2, 2]` and values:

```text
[[22, 28],
 [49, 64]]
```

This calculation explains the recorded [operation](../../glossary.md#operation) meaning. Calling `matmul` does not read the
input values or produce these values eagerly; the current [model](../../glossary.md#term-model) creates [metadata](../../glossary.md#term-metadata) only.

<a id="dynamic-dimensions-and-local-failures"></a>
#### Dynamic [dimensions](../../glossary.md#dimension) and local failures

Both ranks must be at least one. Two unequal static [contraction](../../glossary.md#contraction-dimension) dimensions fail immediately.
When either [contraction dimension](../../glossary.md#contraction-dimension) is unresolved, equality is deferred because it does not affect
the exact result [Shape](../../glossary.md#shape) and both input [descriptors](../../glossary.md#term-descriptor) remain in [provenance](../../glossary.md#provenance). [Batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [broadcasting](../../glossary.md#broadcasting) is
more restrictive because construction must select an exact output Dimension:

- equal unresolved dimensions retain the exact left reference;
- a static [singleton](../../glossary.md#term-singleton) retains the exact opposing reference;
- an unpaired leading [batch dimension](../../glossary.md#batch-dimension) retains its exact input reference;
- an unresolved dimension paired with static non-singleton `S` selects that exact static result
  Dimension and defers the obligation that the unresolved extent bind to <span data-glossary-navigation="true"> ([extent glossary](../../glossary.md#term-extent))</span>`1` or `S`;
- unequal unresolved batch dimensions fail because no exact result Dimension is locally
  derivable; and
- unequal static non-singleton batch dimensions fail.

These rules are local to [MATMUL](../../glossary.md#matrix-multiplication--matmul) and do not change general `ShapeBroadcast`. Null checks, [numeric
promotion](../../glossary.md#numeric-promotion), [rank](../../glossary.md#term-rank) checks, static contraction checks, and leading-to-trailing batch checks all occur
before result [identity](../../glossary.md#term-identity-meanings) allocation. Current package-private [compiler](../../glossary.md#term-compiler) verification proves, rejects,
or retains every deferred equality or singleton-or-equal obligation; concrete [binding](../../glossary.md#term-binding) must
discharge any retained obligation before execution.

<a id="data-type-numerical-meaning-and-result-metadata"></a>
#### [Data type](../../glossary.md#data-type--datatype), numerical meaning, and result [metadata](../../glossary.md#term-metadata)

`DataTypePromotion.promoteNumeric` accepts same-category floating pairs
(including both [FP8](../../glossary.md#term-fp8) types) and same-category signed-integral pairs
([`INT32`](../../glossary.md#data-type--datatype), [`INT64`](../../glossary.md#data-type--datatype)). Unlike low pairs promote to [FLOAT32](../../glossary.md#term-float32); otherwise the promoted type
is the result type. [BOOL](../../glossary.md#data-type--datatype) and floating/integral pairs fail, and construction inserts no cast.

For a [FLOAT64](../../glossary.md#term-float64) result, pairwise products accumulate in FLOAT64. All four low types and FLOAT32
results accumulate in FLOAT32, with one final conversion for a low result. Under
the current FLOAT32 contract, exact broadcast/[contraction](../../glossary.md#contraction-dimension) mapping and contributor membership stay fixed;
every pairwise product participates exactly once under any aggregate tree with per-step FLOAT32
[rounding](../../glossary.md#term-rounding), [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz), and only corresponding multiply/add [fusion](../../glossary.md#term-fusion). The existing qualifying nonempty
final exact-zero [publication](../../glossary.md#term-publication) choice remains local to the complete [MATMUL](../../glossary.md#matrix-multiplication--matmul) cell. Standalone product
and pre-publication arithmetic do not gain that publication rule, a one-term contraction gains no
positive-zero [identity](../../glossary.md#term-identity-meanings) or new [FMA](../../glossary.md#term-fused-multiply-add-fma), and an empty contraction remains [positive zero](../../glossary.md#nan-infinity-and-signed-zero). This adds no
final-output [tolerance](../../glossary.md#term-tolerance), term [loss](../../glossary.md#loss), reduced [precision](../../glossary.md#term-precision), reciprocal substitution, or class change.
Integral products and sums use the promoted signed width modulo `2^32` or `2^64`; an empty
integral contraction produces zero.

Every valid call returns a fresh, unlabeled [Tensor](../../glossary.md#tensor) with the promoted type, exact derived [Shape](../../glossary.md#shape),
unresolved [layout](../../glossary.md#layout), no storage, and [gradient](../../glossary.md#term-gradient) eligibility equal to the logical OR of the operands.
Its one-output [provenance](../../glossary.md#provenance) has index zero and retains `MatmulKind.MATMUL`,
`NoOperationAttrs.INSTANCE`, and exact ordered inputs `[left, right]`. This metadata records a
gradient request, not a gradient rule. It neither [captures](../../glossary.md#term-capture-meanings) or compiles a [graph](../../glossary.md#graph) nor promises
[compiler](../../glossary.md#term-compiler) [validation](../../glossary.md#term-validation), [backend](../../glossary.md#backend) [lowering](../../glossary.md#lowering), a [kernel](../../glossary.md#kernel), storage allocation, or execution.

<a id="linear-projection-convenience"></a>
### [Linear](../../glossary.md#term-linear-meanings)-projection convenience

`input.linear(weight)` and `input.linear(weight, bias)` expose the conventional [parameter](../../glossary.md#term-parameter) [layout](../../glossary.md#layout)
used by a [linear projection](../../glossary.md#linear-projection) without introducing a LINEAR [operation](../../glossary.md#operation). [Weight](../../glossary.md#term-weight) has [Shape](../../glossary.md#shape)
`[outFeatures, inFeatures]`, while [MATMUL](../../glossary.md#matrix-multiplication--matmul) needs its right operand as
`[inFeatures, outFeatures]`. The public convenience is therefore literal composition:

```text
input.linear(weight)       = input.matmul(weight.transpose())
input.linear(weight, bias) = input.matmul(weight.transpose()).add(bias)
```

These receiver conveniences remain on [`Tensor`](../../glossary.md#tensor) because the input naturally owns a single-output
last-[axis](../../glossary.md#term-axis) [contraction](../../glossary.md#contraction-dimension). They own no parameters: the [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) `Linear` [module](../../glossary.md#term-module) is the separate stateful
[owner](../../glossary.md#term-owner) of weight and optional [bias](../../glossary.md#term-bias), while `Tensor.linear` is its stateless [Model](../../glossary.md#term-model) operation
composition. This differs from recurrent [scan](../../glossary.md#term-cumulative-scan)'s multi-input, multi-output domain protocol, which
uses the focused static `RecurrentScan` namespace.

Input [rank](../../glossary.md#term-rank) must be at least one. Its final [Dimension](../../glossary.md#dimension) is `inFeatures`; every preceding input
Dimension is retained in order, and the exact weight axis-zero `outFeatures` Dimension becomes
the final result Dimension. Weight must have rank two. Optional bias is not general [broadcasting](../../glossary.md#broadcasting):
it must have exact rank one, and its sole Dimension must be structurally equal to weight
out-[features](../../glossary.md#term-feature). A null bias is invalid; omit bias by selecting the one-[argument](../../glossary.md#term-argument) overload.

| Input | Weight | Bias | Result |
|---|---|---|---|
| `[K]` | `[N, K]` | absent or `[N]` | `[N]` |
| `[B, K]` | `[N, K]` | absent or `[N]` | `[B, N]` |
| `[B, T, K]` | `[N, K]` | absent or `[N]` | `[B, T, N]` |

Floating pairs and signed-integral pairs inherit current MATMUL promotion. The inner MATMUL also
inherits its family/[dtype](../../glossary.md#data-type--datatype) result contract described above. In the biased form, the product
and bias are promoted again by ordinary ADD, and each visible operation applies only its own named
sites: no cross-[node](../../glossary.md#node) [fusion](../../glossary.md#term-fusion) or algebraic rewrite is introduced. A wider bias may widen only the
final result. The linear conveniences add no numerical freedom, final-output envelope, or inserted
cast. Static unequal contraction Dimensions fail locally; equality
involving an unresolved contraction [extent](../../glossary.md#term-extent) may remain for later [compiler](../../glossary.md#term-compiler) [validation](../../glossary.md#term-validation) or [binding](../../glossary.md#term-binding).
Bias equality never defers: named and expression Dimensions must be structurally equal, and
[identity](../../glossary.md#term-identity-meanings)-based unknowns must be the same unknown.

Every caller-controlled null, promotion, rank, static contraction, and bias failure is validated
before construction, so no partial expression or Tensor ID is created. A successful no-bias call
creates two fresh wrappers and IDs in PERMUTE-then-MATMUL order and returns the MATMUL product
directly. A successful biased call creates three in PERMUTE-then-MATMUL-then-ADD order. Tensor-ID
exhaustion after an earlier intermediate does not roll back that consumed ID.

The no-bias result ends in MATMUL [provenance](../../glossary.md#provenance). The biased result ends in ADD provenance with
ordered inputs `[product, bias]`; the product exposes MATMUL inputs `[input, transposedWeight]`,
and the transposed weight exposes PERMUTE input `[weight]` with axes `[1, 0]`. Every link has one
output at provenance index zero. PERMUTE may preserve a resolved logical [view](../../glossary.md#term-view-meanings) layout, while
MATMUL and ADD results are unresolved, unlabeled, and storage-free. The biased ADD Shape is
structurally equal to the product Shape and reuses its exact ordered Dimension object references,
but the outer Shape objects may differ. No-bias returns the product itself.

<a id="complete-producer-chain-example"></a>
#### Complete [producer](../../glossary.md#term-producer)-chain example

The example constructs a biased [FLOAT32](../../glossary.md#term-float32) projection from input `[2, 3]`, conventional [weight](../../glossary.md#term-weight)
`[4, 3]`, and [bias](../../glossary.md#term-bias) `[4]`, then inspects the primitive producer chain.

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

        // Keep the primitive producer chain visible; linear() does not own the supplied weights.
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

The final two lines distinguish structural [Shape](../../glossary.md#shape) equality from outer Shape [identity](../../glossary.md#term-identity-meanings). Corresponding
[axes](../../glossary.md#term-axis) still reuse the exact input/product [Dimension](../../glossary.md#dimension) references. The example proves [model](../../glossary.md#term-model) [metadata](../../glossary.md#term-metadata)
and ordered producer connectivity only; it does not evaluate numbers, [capture](../../glossary.md#term-capture-meanings) a [compiled graph](../../glossary.md#compiled-graph--compiledgraphmodel),
construct [gradients](../../glossary.md#term-gradient), prove [compiler](../../glossary.md#term-compiler) recognition, or promise [backend](../../glossary.md#backend) [fusion](../../glossary.md#term-fusion) or execution.


---

[Previous topic](windows.md) · [Next topic](attention.md)
