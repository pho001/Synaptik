# Task 0019: Matmul Semantics and [Tensor](../../../../glossary.md#tensor) Expression

**Editorial reading key (current context, not a status update).**

This completed record concerns MATMUL metadata, not execution; its follow-up statuses and
numerical/type scope are historical. The caller invokes `left.matmul(right)`;
`TensorMatmulExpressions` derives the descriptor and delegates result construction to
`TensorFactory`. Ordered operand references and descriptions—type, Shape, and provenance—cross this
boundary, not copied or computed values. The fresh storage-free result owns no result buffer. Model
semantics defines meaning; execution providers realize supported routes rather than redefine it.
Semantic responsibility is not storage ownership.

A contraction sums corresponding components: the example’s first row and first column yield 22
mathematically, not during construction. Batch axes repeat matrix calculations with right-aligned
broadcasting; unresolved extents can leave proof obligations. Follow [matrix-multiplication
expressions](../../../../reference/tensor/matmul-linear.md#matrix-multiplication-expressions) for
rank rules and [current numerical
meaning](../../../../reference/tensor/matmul-linear.md#data-type-numerical-meaning-and-result-metadata)
for expanded low-type coverage and FLOAT32 rounding, fusion, and zero-publication boundaries; these
do not retroactively rewrite this task. Current `Tensor.linear` composes primitives without owning
parameters; the NN `Linear` module separately owns weight and optional bias.

**Mathematical reading illustration.** For real matrices $A\in\mathbb{R}^{m\times k}$ and
$B\in\mathbb{R}^{k\times n}$, with positive integer dimensions $m,k,n$, cell
$C_{ij}=\sum_{r=1}^{k}A_{ir}B_{rj}$ contracts the shared $k$ axis; $1\le i\le m$, $1\le j\le n$.
With $m=n=1$, $k=2$, row $A=[2,3]$ and column $B=[4,5]^\mathsf{T}$, $C_{11}=23$. This explains which
products contribute, not a floating-point reduction order or an execution performed by construction.

## Status

Complete

## Goal

Add one cohesive [backend](../../../../glossary.md#backend)-independent [model](../../../../glossary.md#term-model) contract for [vector](../../../../glossary.md#term-vector), [matrix](../../../../glossary.md#term-matrix-meanings), and batched [matrix
multiplication](../../../../glossary.md#matrix-multiplication--matmul). The task introduces the first-class [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) semantic kind and one public
`Tensor.matmul(Tensor right)` expression while preserving exact [shape](../../../../glossary.md#shape) facts, selected numerical
meaning, and current [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) behavior.

This task records storage-free [metadata](../../../../glossary.md#term-metadata) only. It does not multiply values, choose a matrix
[kernel](../../../../glossary.md#kernel), construct [gradients](../../../../glossary.md#term-gradient), [capture](../../../../glossary.md#term-capture-meanings) or optimize a [graph](../../../../glossary.md#graph), or claim [compiler](../../../../glossary.md#term-compiler), backend, [runtime](../../../../glossary.md#term-runtime), or
end-to-end support.

## Why this is the first split frontier

The former broad task 0019 combined three independently reviewable contracts. [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) is the
primitive needed by both later [linear](../../../../glossary.md#term-linear-meanings) and attention work and fits one [model](../../../../glossary.md#term-model) task. `linear` is an
explicit convenience composition whose [weight](../../../../glossary.md#term-weight) and [bias](../../../../glossary.md#term-bias) contract deserves separate public-[API](../../../../glossary.md#term-application-programming-interface-api)
review. [Scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention) is a named high-level semantic [operation](../../../../glossary.md#operation) with distinct
masking, causal, scale, and numerical policies. Combining all three would exceed normal [planning](../../../../glossary.md#term-planning)
granularity and make attention's later decisions block the primitive.

Task 0019 therefore owns MATMUL only. Tasks 0019D and 0019E remain Draft follow-ups without
detailed specifications. Existing tasks 0019A–0019C keep their stable identifiers and order.

## [Mental model](../../../../glossary.md#term-mental-model) and examples

[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) treats the last two [axes](../../../../glossary.md#term-axis) of each [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)-or-greater operand as [matrix](../../../../glossary.md#term-matrix-meanings) axes. A [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
left operand is temporarily treated as `[1, K]` and that inserted result axis is removed. A
rank-one right operand is temporarily treated as `[K, 1]` and that inserted result axis is
removed. Leading axes are [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) axes and broadcast right-aligned.

```text
[K]       @ [K]          -> []
[M, K]    @ [K]          -> [M]
[K]       @ [K, N]       -> [N]
[M, K]    @ [K, N]       -> [M, N]
[2, M, K] @ [1, K, N]    -> [2, M, N]
[K]       @ [B, K, N]    -> [B, N]
[B, M, K] @ [K]          -> [B, M]
```

For concrete rank-two values:

```text
[[1, 2, 3],      [[1, 2],       [[22, 28],
 [4, 5, 6]]   @   [3, 4],    =   [49, 64]]
                  [5, 6]]
```

Each output is the [dot product](../../../../glossary.md#term-dot-product) of one left row and one right column. The example defines
mathematical meaning; this [model](../../../../glossary.md#term-model) task does not evaluate it.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly `MatmulKind.MATMUL` in a new [linear](../../../../glossary.md#term-linear-meanings)-algebra [operation](../../../../glossary.md#operation) package.
- Pair it only with `NoOperationAttrs.INSTANCE` and a fixed two-input, one-output [signature](../../../../glossary.md#term-signature).
- Add exactly `public Tensor matmul(Tensor right)`.
- Add one package-private, field-free `TensorMatmulExpressions` construction helper.
- Accept [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and higher operands under the exact [vector](../../../../glossary.md#term-vector)/[matrix](../../../../glossary.md#term-matrix-meanings)/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) rules below.
- Derive exact static, named-symbolic, and expression-dimension result Shapes without runtime
  binding or graph-wide constraint solving.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [Shapes glossary](../../../../glossary.md#shape); [runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding); [graph glossary](../../../../glossary.md#graph))</span>
- Accept same-category floating or signed-integral pairs through `DataTypePromotion.promoteNumeric`;
  reject BOOL and cross-category pairs without implicit casts.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Fix result, accumulation, [overflow](../../../../glossary.md#term-overflow), empty-[contraction](../../../../glossary.md#contraction-dimension), and determinism policies below.
- Construct one fresh unresolved-layout result with exact ordered producer inputs and no label or
  storage.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [producer glossary](../../../../glossary.md#term-producer))</span>
- Add focused semantic and public-expression tests.
- Finalize Javadocs, Tensor/Compile API references, glossary, capabilities, and planning records
  through a mandatory separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [`linear`](../../../../glossary.md#term-linear-meanings), scaled dot-product attention, attention weights, masks, causal behavior, dropout, RNG,
  sorting, top-K, activation, embedding, convolution, pooling, normalization, or losses<span data-glossary-navigation="true"> ([scaled dot-product attention glossary](../../../../glossary.md#scaled-dot-product-attention); [weights glossary](../../../../glossary.md#term-weight); [masks glossary](../../../../glossary.md#term-boolean-mask); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [activation glossary](../../../../glossary.md#term-activation); [embedding glossary](../../../../glossary.md#embedding); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [normalization glossary](../../../../glossary.md#term-normalization); [losses glossary](../../../../glossary.md#loss))</span>
- transpose flags, alpha/beta coefficients, output data-type overrides, accumulator options,
  sparse/quantized/complex/unsigned values, FLOAT16, or general equation notation<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [accumulator glossary](../../../../glossary.md#accumulator-type); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
- [eager](../../../../glossary.md#term-eager-evaluation) value reads, [host](../../../../glossary.md#term-host)-storage allocation, result storage, resolved result [layout](../../../../glossary.md#layout), or mutation
- compiler capture, graph-wide shape constraints, canonicalization, decomposition, fusion,
  autograd, gradient formulas, or backward operations<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [autograd glossary](../../../../glossary.md#autograd); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- capability providers, backend lowering, Basic Linear Algebra Subprograms (BLAS), kernels,
  tolerances, runtime execution, conformance, or integration support<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [Basic Linear Algebra Subprograms glossary](../../../../glossary.md#mps-mpsgraph-and-openblas); [BLAS glossary](../../../../glossary.md#term-blas); [kernels glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to [`DataType`](../../../../glossary.md#data-type--datatype), `DataTypePromotion`, [`Dimension`](../../../../glossary.md#dimension), `DimensionExpressions`, [`Shape`](../../../../glossary.md#shape),
  `ShapeBroadcast`, `TensorDescriptor`, producer/provenance/factory contracts, Gradle, another
  module, <span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>`ARCHITECTURE.md`, or focused architecture documentation

## Exact semantic and [API](../../../../glossary.md#term-application-programming-interface-api) contract

### Kind, attributes, and [signature](../../../../glossary.md#term-signature)

Create exactly:

```java
public enum MatmulKind implements OperationKind {
    MATMUL
}
```

[`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) declares one stable signature:

```java
OperationSignature.fixed(NoOperationAttrs.class, 2, 1)
```

It accepts only `NoOperationAttrs.INSTANCE`. Add no matmul attributes, [transpose](../../../../glossary.md#term-transpose) variants, [aliases](../../../../glossary.md#aliasing-and-arena),
string [dispatch](../../../../glossary.md#term-dispatch), [registry](../../../../glossary.md#term-registry), [backend](../../../../glossary.md#backend) support, or execution [metadata](../../../../glossary.md#term-metadata).

Add exactly this public method:

```java
public Tensor matmul(Tensor right)
```

The receiver is the ordered left operand. The exact [producer](../../../../glossary.md#term-producer) input order is `[this, right]`.

### [Rank](../../../../glossary.md#term-rank) and [matrix](../../../../glossary.md#term-matrix-meanings)-[axis](../../../../glossary.md#term-axis) rules

Both operands must have rank at least one. Define:

- left [contraction](../../../../glossary.md#contraction-dimension) axis: the final left axis;
- right contraction axis: the final right axis for rank one, otherwise the penultimate right axis;
- left row axis `M`: absent for rank one, otherwise the penultimate left axis;
- right column axis `N`: absent for rank one, otherwise the final right axis;
- left [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) prefix: every left axis before its final two axes, or empty for rank one;
- right batch prefix: every right axis before its final two axes, or empty for rank one.

Broadcast the batch prefixes right-aligned. Form the output in this order:

1. broadcast batch [dimensions](../../../../glossary.md#dimension);
2. append `M` when the left operand is not rank one;
3. append `N` when the right operand is not rank one.

Two [vectors](../../../../glossary.md#term-vector) therefore produce `Shape.scalar()`. Every retained `M`, `N`, equal batch, [singleton](../../../../glossary.md#term-singleton)-
expanded opposing, and unpaired leading `Dimension` uses the exact input reference.

### Inner-[dimension](../../../../glossary.md#dimension) compatibility

If both [contraction](../../../../glossary.md#contraction-dimension) dimensions are static, their sizes must be equal. Reject a mismatch before
[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) processing. Zero is valid when both are statically zero and denotes an empty contraction.

If either [contraction dimension](../../../../glossary.md#contraction-dimension) is unresolved, accept the request unless a static mismatch is
already proven. Structurally equal named or expression dimensions are accepted directly. Unequal
named, expression, static/unresolved, or unresolved/unresolved pairs [carry](../../../../glossary.md#term-carry) a deferred equality
obligation: [compiler](../../../../glossary.md#term-compiler) [shape](../../../../glossary.md#shape) [validation](../../../../glossary.md#term-validation) or later [binding](../../../../glossary.md#term-binding) must prove equal concrete sizes before
execution. No new constraint object is added because the contraction [extent](../../../../glossary.md#term-extent) does not occur in the
result Shape and both exact input [descriptors](../../../../glossary.md#term-descriptor) remain available through ordered [provenance](../../../../glossary.md#provenance).

### [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [broadcasting](../../../../glossary.md#broadcasting) and deferred obligations

For each aligned batch pair:

- structurally equal [dimensions](../../../../glossary.md#dimension) retain the exact left reference;
- a static [singleton](../../../../glossary.md#term-singleton) retains the exact opposing reference;
- an unpaired [leading dimension](../../../../glossary.md#term-leading-dimension) retains its exact input reference;
- two unequal static non-singletons are rejected;
- one unresolved dimension paired with a static non-singleton accepts the static dimension as the
  exact result extent and defers the obligation that the unresolved extent bind to either one or
  that static size; and<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- two unequal unresolved dimensions are rejected because the current model cannot derive one
  exact result extent without inventing a non-deterministic unknown dimension.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

This is deliberately local [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) algebra. Do not broaden or change `ShapeBroadcast`.

Examples using `N`, `B`, and `C` as distinct named dimensions:

```text
[N, M, K] @ [1, K, P] -> [N, M, P]
[N, M, K] @ [4, K, P] -> [4, M, P] with deferred N in {1, 4}
[B, M, K] @ [B, K, P] -> [B, M, P]
[B, M, K] @ [C, K, P] -> rejected: exact batch result is not locally derivable
```

The same structural rules apply to `ExpressionDimension` values.

### [Data type](../../../../glossary.md#data-type--datatype), result, and numerical policy

Use `DataTypePromotion.promoteNumeric(leftType, rightType)` unchanged:

- [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), and [FLOAT64](../../../../glossary.md#term-float64) pairs promote through the current floating hierarchy;
- [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) pairs promote through the current signed-integral hierarchy;
- floating/integral pairs and every [BOOL](../../../../glossary.md#data-type--datatype) pair are rejected; and
- no cast [producer](../../../../glossary.md#term-producer) is inserted.

The promoted type is the result type. Floating accumulation uses FLOAT64 for a FLOAT64 result,
FLOAT32 for a FLOAT32 result, and FLOAT32 for a BFLOAT16 result followed by conversion to
BFLOAT16 output. The portable [target](../../../../glossary.md#term-target) is the sum of pairwise products over the [contraction](../../../../glossary.md#contraction-dimension) [extent](../../../../glossary.md#term-extent).
[Backends](../../../../glossary.md#backend) may reassociate terms and use [fused multiply-add](../../../../glossary.md#term-fused-multiply-add-fma), so bitwise equality, one traversal
order, and cross-backend identical [rounding](../../../../glossary.md#term-rounding) are not promised; later backend [conformance](../../../../glossary.md#term-conformance) must set
per-type error tolerances. [IEEE](../../../../glossary.md#term-ieee)-754 [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) behavior follows the selected
floating multiply/add operations. An empty floating contraction produces [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero).

Integral multiplication and accumulation [mean](../../../../glossary.md#term-mean) the exact sum of products modulo `2^32` or `2^64`
in the promoted signed [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks) result type. No [widening](../../../../glossary.md#term-widening), [saturation](../../../../glossary.md#term-saturation), [overflow](../../../../glossary.md#term-overflow) exception,
or undefined overflow is selected. Modular [reassociation](../../../../glossary.md#term-reassociation) is permitted and exact. An empty
integral contraction produces zero.

These are semantic policies, not a [kernel](../../../../glossary.md#kernel), instruction, [BLAS](../../../../glossary.md#term-blas), loop, or storage design.

### Result [metadata](../../../../glossary.md#term-metadata) and [provenance](../../../../glossary.md#provenance)

Every successful call creates:

- one fresh `TensorDescriptor` with promoted result type, exact derived [Shape](../../../../glossary.md#shape),
  `Optional.empty()` [layout](../../../../glossary.md#layout), and `requiresGrad` equal to the logical OR of both inputs;
- one `Operation(MATMUL, NoOperationAttrs.INSTANCE)`;
- one fresh single-output [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) retaining exact ordered inputs `[left, right]` and the
  exact result descriptor;
- one fresh unlabeled, storage-free [Tensor](../../../../glossary.md#tensor) with output index zero and one newly allocated ID.

The inputs, their [descriptors](../../../../glossary.md#term-descriptor), labels, storage, provenance, and IDs remain unchanged. Repeated
valid calls create distinct [producer](../../../../glossary.md#term-producer) and Tensor [identities](../../../../glossary.md#term-identity-meanings) even when operands are identical.

### [Validation](../../../../glossary.md#term-validation) order, failures, and ID effects

`TensorMatmulExpressions.apply(left, right)` performs this exact order:

1. null-check `left`, then `right`, with the [parameter](../../../../glossary.md#term-parameter) name as message;
2. call `DataTypePromotion.promoteNumeric` on [descriptor](../../../../glossary.md#term-descriptor) types;
3. require left [rank](../../../../glossary.md#term-rank) at least one, then right rank at least one;
4. reject a proven static [contraction](../../../../glossary.md#contraction-dimension) mismatch;
5. process aligned [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [dimensions](../../../../glossary.md#dimension) from leading result batch [axis](../../../../glossary.md#term-axis) to trailing;
6. build the result [Shape](../../../../glossary.md#shape), descriptor, and [operation](../../../../glossary.md#operation); and
7. delegate exactly once to `TensorFactory.createDerived`.

Use exact task-owned messages:

```text
left rank must be at least 1: 0
right rank must be at least 1: 0
matmul inner dimensions must match: left=<dimension>, right=<dimension>
cannot derive exact matmul batch dimension at result batch axis <axis>: left=<dimension>, right=<dimension>
cannot broadcast matmul batch dimensions at result batch axis <axis>: left=<dimension>, right=<dimension>
```

The first batch message is for unequal unresolved dimensions. The second is for unequal static
non-singletons. Existing `DataTypePromotion` messages remain unchanged.

Every failure through step 6 consumes no [Tensor](../../../../glossary.md#tensor) ID. Only the final factory delegation may allocate
an ID; exhaustion keeps the existing exact message and behavior. Tests must inspect the shared ID
counter only for this concrete no-ID-on-local-failure risk.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)

## Architecture constraints

- Work remains entirely inside `modules/model` plus its documentation/[planning](../../../../glossary.md#term-planning) records.
- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and is not [graph](../../../../glossary.md#graph) [IR](../../../../glossary.md#term-intermediate-representation-ir).
- [`Operation`](../../../../glossary.md#operation) and [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) record [backend](../../../../glossary.md#backend)-independent meaning only and expose no support or route.
- Package direction remains `model.tensor -> model.operation.linalg`, datatype, and shape; the
  operation package must not import <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>`Tensor` or graph state.
- Compiler later owns capture, graph-wide revalidation, symbolic binding, canonicalization, and
  gradients. Backend prepare owns lowering, algorithms, fusion, and kernel/BLAS route choice.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [gradients glossary](../../../../glossary.md#term-gradient); [prepare glossary](../../../../glossary.md#prepare); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel); [BLAS glossary](../../../../glossary.md#term-blas))</span>
- No architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), [module](../../../../glossary.md#term-module)-boundary, [Gradle](../../../../glossary.md#term-gradle), or cross-module change.

## Legacy evidence

The read-only `legacy/pre-rewrite` branch confirms workloads for [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and batched [MATMUL](../../../../glossary.md#matrix-multiplication--matmul),
[linear](../../../../glossary.md#term-linear-meanings) projections, attention products, [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) execution, and [backend route](../../../../glossary.md#backend-route)
selection. It also mixed front-end [validation](../../../../glossary.md#term-validation) with [backend](../../../../glossary.md#backend) result-type decisions and contains
architecture, package, [lowering](../../../../glossary.md#lowering), execution, and optimization structures that are not carried
forward. It is test and workload evidence only, not design authority. This task is implemented
from current contracts.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.operation`

Package added:

- `io.github.pho001.synaptik.model.operation.linalg` — public backend-independent linear-algebra
  semantic kinds; no Tensor, compiler, backend, or execution behavior.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [linear glossary](../../../../glossary.md#term-linear-meanings); [Tensor glossary](../../../../glossary.md#tensor); [compiler glossary](../../../../glossary.md#term-compiler))</span>

Type placement:

- `io.github.pho001.synaptik.model.operation.linalg.MatmulKind` — public typed MATMUL identity and
  family-owned fixed signature.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [identity glossary](../../../../glossary.md#term-identity-meanings); [signature glossary](../../../../glossary.md#term-signature))</span>
- `io.github.pho001.synaptik.model.tensor.TensorMatmulExpressions` — package-private local shape,
  type, descriptor, operation, and provenance construction boundary.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.model.tensor.Tensor` — existing public fluent [API](../../../../glossary.md#term-application-programming-interface-api) [owner](../../../../glossary.md#term-owner).

Tests mirror the production package whose package-private behavior they inspect.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/linalg/MatmulKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/linalg/MatmulSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — update only
  the exact public Tensor API inventory and method count for <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>`matmul`.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java` —
  update only the shared exact public Tensor method count.

Documentation/[planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless the implementation makes an existing claim inaccurate: Training [API](../../../../glossary.md#term-application-programming-interface-api);
[DataType](../../../../glossary.md#data-type--datatype)/[Shape](../../../../glossary.md#shape)/TensorDescriptor/[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance)/factory and related expression contracts;
architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

At most three production, four test, and seven documentation/[planning](../../../../glossary.md#term-planning) files: exactly 14 paths.
`Tensor.java` changes only for the exact method, imports, and its [Javadoc](../../../../glossary.md#term-javadoc). The task adds no
attribute type and changes no existing source or test contract except the two explicitly listed
public [API](../../../../glossary.md#term-application-programming-interface-api) inventory/count updates required by the new method. Stop for a fifteenth path, another
production/test type, any other existing test edit, cross-[module](../../../../glossary.md#term-module) work, or architecture change.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Document MATMUL's vector promotion/removal mental model, exact matrix axes, batch broadcasting,
  contraction, dynamic obligations, types, numerical policies, metadata, provenance, and layer
  boundaries.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [vector glossary](../../../../glossary.md#term-vector); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [axes glossary](../../../../glossary.md#term-axis); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [broadcasting glossary](../../../../glossary.md#broadcasting); [contraction glossary](../../../../glossary.md#contraction-dimension); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Document helper [validation](../../../../glossary.md#term-validation) order, exact failures, construction order, and ID effects.
- Document `Tensor.matmul` parameters, result, Shape/type/gradient/layout/storage/provenance,
  freshness, failures, deferred constraints, and lack of execution/gradient/backend support.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout); [backend glossary](../../../../glossary.md#backend))</span>
- Update [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) with the full [rank](../../../../glossary.md#term-rank)/[Shape](../../../../glossary.md#shape) table and at least one concrete mathematical example.
- Update Compile API only to keep current model-expression versus planned capture boundaries
  accurate; do not claim compiler support.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Add or revise glossary terms only when reusable terminology is introduced. At minimum review
  MATMUL, contraction dimension, batch dimension, and matrix multiplication.<span data-glossary-navigation="true"> ([contraction dimension glossary](../../../../glossary.md#contraction-dimension); [batch dimension glossary](../../../../glossary.md#batch-dimension); [matrix multiplication glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Keep capabilities, master plan, task, and roadmap synchronized. Record reasoned no-change
  conclusions for reviewed documents and contracts.

## Acceptance criteria

- Exactly one [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) kind exists with exact [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs), two-input, one-output [signature](../../../../glossary.md#term-signature).
- Exactly one public `matmul(Tensor)` method is added; the public [Tensor](../../../../glossary.md#tensor) method count becomes 157.
- Rank-one through batched combinations derive every specified output and preserve exact retained
  Dimension references.<span data-glossary-navigation="true"> ([Rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Dimension glossary](../../../../glossary.md#dimension))</span>
- Static and symbolic/expression inner and batch compatibility follows the exact immediate versus
  deferred policy without changing <span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>`ShapeBroadcast` or inventing result constraints.
- All floating and integral ordered width pairs promote and record the selected result and
  accumulation/overflow policies; BOOL and cross-category pairs fail.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- [Validation](../../../../glossary.md#term-validation) order, exact task-owned messages, and no-ID local failures match the specification.
- Results are fresh, storage-free, unlabeled, layout-unresolved, use gradient-request OR, and have
  exact operation/input/descriptor/output-index provenance; inputs remain unchanged.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Tests validate contracts and [metadata](../../../../glossary.md#term-metadata) only; they do not pretend to execute MATMUL values.
- No linear, attention, dropout, gradient, compiler, backend, runtime, registry, dependency,
  Gradle, architecture, or other-module work lands.<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [registry glossary](../../../../glossary.md#term-registry); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>
- Focused tests, exactly one final model suite after Java stability, Javadoc, documentation, link,
  exact 14-path scope, status, formatting, and whitespace validation pass.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- A separate clean-context documentation pass finalizes all authorized documentation and records
  reused Java evidence plus no-change conclusions.
- Task 0019 becomes Complete only after all evidence is recorded; 0018V remains Complete; 0019D,
  0019E, existing 0019A–0019C, and every later task remain Draft without detailed specs.

## Tests / [validation](../../../../glossary.md#term-validation)

Required focused command:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.linalg.MatmulSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.shape.ShapeBroadcastTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [model](../../../../glossary.md#term-model) suite:

```bash
./gradlew :modules:model:test
```

The focused tests cover exact enum/[signature](../../../../glossary.md#term-signature)/helper/public surfaces; all [rank](../../../../glossary.md#term-rank) families; static,
named, expression, [singleton](../../../../glossary.md#term-singleton), zero, and deferred [Shape](../../../../glossary.md#shape) cases; type pairs; validation order/messages
and ID effects; [descriptor](../../../../glossary.md#term-descriptor)/[provenance](../../../../glossary.md#provenance)/freshness; and the numerical contract as [metadata](../../../../glossary.md#term-metadata) without
[eager evaluation](../../../../glossary.md#term-eager-evaluation).

The separate documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also validate local Markdown links and anchors, balanced fences, terminology, generated Javadoc,
final newlines, trailing whitespace, exact 14 paths, package placement, public method count,
synchronized statuses, one [Ready](../../../../glossary.md#term-ready) frontier, 0018V Complete, all follow-ups Draft, and no unintended
later detailed specification. Manual reflection/source checks are justified only when the focused
automated surface tests do not cover a concrete risk.

Repository-wide validation is deferred to the selected-modern-operations [checkpoint](../../../../glossary.md#term-checkpoint) after task
0022 or [CI](../../../../glossary.md#term-continuous-integration-ci). This single-[module](../../../../glossary.md#term-module) task changes no [dependency](../../../../glossary.md#term-dependency), architecture boundary, shared [Gradle](../../../../glossary.md#term-gradle)
contract, or other module, and must not duplicate the completed 0018V repository checkpoint.

## Dependencies

- 0001–0002 and 0018M: current [DataType](../../../../glossary.md#data-type--datatype) promotion, [Dimensions](../../../../glossary.md#dimension), [Shapes](../../../../glossary.md#shape), and symbolic extents.
- 0005–0007, 0011–0013, and 0018K–0018L: operation signatures, descriptors, Tensor, IDs, and
  shared producer/provenance construction.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature); [descriptors glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- 0018N: exact typed-value boundary to preserve unchanged.
- 0018T, 0018U, and 0018V: selected same-category promotion and numerical-policy conventions.

## Follow-up tasks

- 0019D remains Draft for the exact [`linear`](../../../../glossary.md#term-linear-meanings) convenience composition over MATMUL, transpose, and
  optional bias addition.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [transpose glossary](../../../../glossary.md#term-transpose); [bias glossary](../../../../glossary.md#term-bias))</span>
- 0019E remains Draft for one-output scaled dot-product attention without dropout or public
  attention-weight output.<span data-glossary-navigation="true"> ([scaled dot-product attention glossary](../../../../glossary.md#scaled-dot-product-attention); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [weight glossary](../../../../glossary.md#term-weight))</span>
- [Compiler](../../../../glossary.md#term-compiler) later owns [capture](../../../../glossary.md#term-capture-meanings), deferred constraint proof, [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), and [autograd](../../../../glossary.md#autograd).
- Backend/conformance/runtime/integration work later owns algorithms, tolerances, lowering,
  kernels, storage, and execution.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [runtime glossary](../../../../glossary.md#term-runtime); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel))</span>

Do not create another detailed task specification during task 0019 implementation.

## Architecture impact

Expected impact: None.

If implementation requires an architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), focused-architecture,
cross-[module](../../../../glossary.md#term-module), or [scope](../../../../glossary.md#term-scope) change, stop and report the conflict instead of editing around it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in Synaptik without commit or push. Read AGENTS.md, ARCHITECTURE.md, current architecture,
documentation/planning rules, roadmap, model capabilities/master plan, completed foundation tasks
0001–0003, 0005–0008, 0013, relevant expression/softmax/reduction tasks, reset tasks 0018K–0018V,
and task 0019.

Implement docs/planning/modules/model/tasks/0019-matmul-semantics-and-tensor-expression.md exactly
inside its 14 authorized paths. The two added existing test paths may change only for the exact
public Tensor API inventory/count from 156 to 157 and the one new matmul method as applicable.
Preserve current contracts. Stop on architecture uncertainty, scope overflow, another
type/test/document need, or cross-module work.

Run focused validation and exactly one final model suite after Java stabilizes. Hand the actual
diff and exact Java evidence to a separate clean-context documentation agent in the same overall
change. That agent finalizes Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and
documentation checks while reusing successful Java evidence. Synchronize status only after all
criteria pass; keep 0018V Complete and every follow-up Draft without detailed specs.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Give the separate clean-context documentation agent this task, the complete implementation diff,
exact focused/final [model](../../../../glossary.md#term-model) evidence and whether Java changed afterward, all selected [Shape](../../../../glossary.md#shape)/type/
numerical/[provenance](../../../../glossary.md#provenance) policies, the seven authorized documentation paths, and required [Javadoc](../../../../glossary.md#term-javadoc),
Markdown, [scope](../../../../glossary.md#term-scope), and status [validation](../../../../glossary.md#term-validation).

The documentation agent independently reads AGENTS, architecture, documentation rules and the
General/[API](../../../../glossary.md#term-application-programming-interface-api)-Javadoc/[Planning](../../../../glossary.md#term-planning)/Example profiles, this task, actual source/tests/generated Javadoc,
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs, glossary, capabilities/master/roadmap, and directly related
[DataType](../../../../glossary.md#data-type--datatype)/Shape/[operation](../../../../glossary.md#operation)/provenance contracts. It finalizes Javadocs and documentation, checks the
mathematical example, and records reasoned no-change conclusions for Training API, foundational
contracts, architecture/ADRs/tests, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), other [modules](../../../../glossary.md#term-module), and follow-ups.

It does not repeat successful Java tests unless executable Java changes, evidence is stale, or a
concrete recorded risk requires a rerun. It records the clean-context identifier, reused evidence,
files/topics reviewed, commands/results, glossary impact, limitations, and unresolved issues.

## Local decisions

- Kept [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)'s [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) algebra in the field-free package-private helper instead of changing shared
  `ShapeBroadcast`: MATMUL may defer an unresolved-versus-static singleton-or-equal obligation
  only when the exact static output extent is locally derivable.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [extent glossary](../../../../glossary.md#term-extent))</span>
- Retained exact input Dimension references wherever the result selects an existing row, column,
  equal batch, singleton-opposing, or unpaired batch extent. No new constraint or synthetic
  unknown Dimension is created.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension))</span>
- Used the existing `DataTypePromotion.promoteNumeric` boundary unchanged. Result and accumulator
  policy is documentation-level semantic meaning; model expression construction performs no
  numerical work.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [model glossary](../../../../glossary.md#term-model))</span>
- Used direct helper delegation from `Tensor.matmul` so the package-private helper surface remains
  exactly its two authored methods; this avoids a compiler-generated synthetic helper method that
  would violate the focused surface contract.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- Kept tasks 0019A–0019E and all later work Draft without detailed specifications. Task 0019A is
  the next Draft planning frontier after this completed task.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Known limitations

- Unequal unresolved [contraction](../../../../glossary.md#contraction-dimension) [dimensions](../../../../glossary.md#dimension) are representable but require later equality proof.
- An unresolved batch dimension can pair with a known non-singleton extent only under a deferred
  singleton-or-equal obligation.<span data-glossary-navigation="true"> ([batch dimension glossary](../../../../glossary.md#batch-dimension); [singleton glossary](../../../../glossary.md#term-singleton); [extent glossary](../../../../glossary.md#term-extent))</span>
- Two unequal unresolved batch dimensions are rejected because no exact result Dimension can be
  selected locally.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- [Model](../../../../glossary.md#term-model) completion does not imply numeric execution or [gradient](../../../../glossary.md#term-gradient) support.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0019_implementation` ran the required focused command after
  executable Java stabilized:
  `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.linalg.MatmulSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.shape.ShapeBroadcastTest`.
  It passed 32 tests with zero failures and zero errors.
- The same implementation context then ran exactly one final `./gradlew :modules:model:test`.
  It passed 758 tests across 93 suites with zero failures and zero errors. An earlier focused run
  had exposed a compiler-generated synthetic method; the implementation replaced the method
  reference before both successful recorded runs. Executable Java did not change after them.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Documentation context `/root/task_0019_implementation/task_0019_documentation` reused that Java
  evidence as required and did not rerun Java tests because it changed only Javadocs and Markdown.
  It independently reviewed all three production files, all four tests, generated Javadoc, Tensor,
  Compile, and Training API references, glossary, model capabilities/master plan, roadmap, and the
  directly related data-type promotion, Shape/Dimension, operation signature, descriptor,
  factory, producer, and provenance contracts.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [operation signature glossary](../../../../glossary.md#operation-signature); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Documentation context ran `./gradlew :modules:model:javadoc` after the final Javadoc edits;
  it passed with two executed tasks. Generated pages for `MatmulKind` and `Tensor.matmul` contain
  the final semantic and accumulator text. The package-private helper is intentionally absent from
  public generated Javadoc and was reviewed directly in source.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type))</span>
- A targeted Ruby Markdown checker resolved 505 local links, including 145 heading anchors, across
  all seven changed documentation/planning files with zero errors. Two preliminary checker
  invocations failed in the validation script itself because the installed Ruby lacks<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
  `filter_map` and because the first heading-slug approximation mishandled slash-separated
  headings; the corrected checker then passed without requiring documentation changes.
- A targeted Ruby formatting check passed for all seven Markdown files: balanced backtick and
  tilde fences, final newlines, and no trailing whitespace. `git diff --check` passed.
- Exact-scope and source/generated-surface checks passed: exactly 14 changed or untracked paths;
  the three production, four test, and seven documentation/planning files match the authorized
  list; both new types match their planned packages; compiled <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>[`Tensor`](../../../../glossary.md#tensor) has exactly 157 public
  method entries and no public constructor; and generated Javadoc contains `MatmulKind` and
  `Tensor.matmul`.
- Status checks confirmed task 0018V remains Complete, task 0019 and its master/roadmap rows are
  Complete, and tasks 0019A–0019E plus every later model task remain Draft. No detailed
  0019A–0019E specification exists; task 0019A is the next Draft frontier.
- The [matrix](../../../../glossary.md#term-matrix-meanings) example was recalculated exactly as four integer [dot products](../../../../glossary.md#linear-algebra-building-blocks): `22`, `28`, `49`, and
  `64`. Terminology checks found the three new glossary entries and consistent uses in Tensor API
  and Javadoc. A preliminary generated-page inspection used one incorrect output path; inspection
  against the actual generated paths then passed. No product file was changed by these validation
  script corrections.

## Implementation notes

- Added public `MatmulKind.MATMUL` with its sole [immutable](../../../../glossary.md#term-immutable) fixed
  [`NoOperationAttrs`](../../../../glossary.md#nooperationattrs)/two-input/one-output [signature](../../../../glossary.md#term-signature).
- Added the package-private field-free `TensorMatmulExpressions` helper and public
  `Tensor.matmul(Tensor right)`. Construction implements every vector/matrix/batch rank family,
  selected immediate and deferred Shape obligations, same-category numeric promotion, exact
  metadata/provenance, validation order/messages, and no-ID local failure boundary.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [numeric promotion glossary](../../../../glossary.md#numeric-promotion); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation))</span>
- Added focused semantic and expression tests, and changed only the authorized Tensor public API
  inventory/count assertions in the two existing tests. The public Tensor method count is 157.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Finalized all three production Javadocs. Expanded the Tensor API with the complete rank/Shape
  table, exact dynamic obligations, numeric and metadata policies, and the independently checked
  `[[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]` example. Updated Compile API only
  to distinguish current model expressions from planned capture and validation.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Added glossary entries for batch dimension, contraction dimension, and matrix
  multiplication/MATMUL. These are reusable terms introduced by the public contract.<span data-glossary-navigation="true"> ([batch dimension glossary](../../../../glossary.md#batch-dimension); [contraction dimension glossary](../../../../glossary.md#contraction-dimension); [matrix multiplication glossary](../../../../glossary.md#matrix-multiplication--matmul); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Reviewed Training API with no change: task 0019 adds no gradient rule, trainable role, or
  training workflow. Reviewed foundational datatype, Shape/Dimension, operation,
  TensorDescriptor/factory/producer/provenance contracts with no change because MATMUL composes
  them without altering their behavior. Architecture contract, focused architecture pages, ADRs,
  architecture tests, backend conformance, integration tests, Gradle, other modules, and Draft
  follow-ups also require no change because this task adds model metadata only and changes no
  boundary, dependency, backend/runtime behavior, or cross-module contract.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [Dimension glossary](../../../../glossary.md#dimension); [operation glossary](../../../../glossary.md#operation); [producer glossary](../../../../glossary.md#term-producer); [Architecture contract glossary](../../../../glossary.md#architecture-contract); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Completion summary

- Completed changes: first-class MATMUL semantics, local vector/matrix/batch expression
  construction, public Tensor method, focused tests, complete Javadocs/API reference, glossary,
  capabilities, and synchronized planning records.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [vector glossary](../../../../glossary.md#term-vector); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly the three production, four test, and seven
  documentation/planning paths listed under Affected files.
- Tests and validation: reused the successful 32-test focused run and 758-test/93-suite final model
  run from the implementation context; final Javadoc, Markdown, scope, status, and whitespace
  checks are recorded in Validation evidence.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context
  `/root/task_0019_implementation/task_0019_documentation` independently finalized the authorized
  documentation and Javadocs in the same overall change.
- Documentation impact: Tensor API, Compile API, glossary, capabilities, task, model master plan,
  and roadmap now describe current MATMUL metadata and preserve planned compiler/backend/runtime
  boundaries. Training API and other reviewed areas require no change for the reasons above.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: all three production paths document meaning, parameters/results, failure
  conditions, dynamic obligations, numerical policy, provenance, ID effects, and unsupported
  layers without claiming execution.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [provenance glossary](../../../../glossary.md#provenance))</span>
- Glossary impact: added reusable definitions for batch dimension, contraction dimension, and
  matrix multiplication/MATMUL.<span data-glossary-navigation="true"> ([batch dimension glossary](../../../../glossary.md#batch-dimension); [contraction dimension glossary](../../../../glossary.md#contraction-dimension); [matrix multiplication glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Known limitations: the accepted deferred obligations and metadata-only boundary remain as
  listed above.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
