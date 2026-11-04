# Task 0019E: [Scaled Dot-Product Attention](../../../../glossary.md#scaled-dot-product-attention)

**Editorial reading key (current context, not a status update).**

This record concerns the original one-output attention construction, not today’s broader compiler or
attention surface. The caller uses query as the Tensor receiver and supplies key, value, optional
BOOL eligibility mask, and scale/causal attributes; the construction helper validates these
descriptions and delegates to the Tensor factory. It records ordered input references and result
metadata, not computed values. These storage-free expressions neither execute attention nor own
result buffers. Model semantics defines meaning; compiler proofs and execution-provider routes do
not redefine it. Semantic responsibility is not storage ownership.

A score measures query/key similarity; softmax turns eligible scores into weights for combining
value rows. For example, at the first query position, causal selection admits only the first key,
subject also to the explicit mask. Exclusion precedes arithmetic: an excluded NaN value is ignored,
whereas an eligible zero-weight NaN need not be. Follow [current attention
expressions](../../../../reference/tensor/attention.md#scaled-dot-product-attention-expressions) to
distinguish the original output-only calls from later weights-returning calls, and [eligibility
rules](../../../../reference/tensor/attention.md#eligibility-and-special-values) for the
special-value boundaries behind the historical schematic.

**Mathematical reading illustration.** For two eligible finite real scores $s_1=s_2=0$, define
weights $w_j=\exp(s_j)/(\exp(s_1)+\exp(s_2))$ for key position $j\in\{1,2\}$. For scalar values
$v_1=2$, $v_2=6$, the weighted output $y=w_1v_1+w_2v_2=4$ because both weights are one half. This
real-number example assumes a nonempty eligible set; it does not cover excluded values, all-masked
rows, NaN/infinity, or prescribe floating evaluation.

## Status

Complete

## Goal

Add one [backend](../../../../glossary.md#backend)-independent, one-output [scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention) semantic [operation](../../../../glossary.md#operation) and its
public [`Tensor`](../../../../glossary.md#tensor) construction. The receiver is query. Key, value, an optional [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask), causal
selection, and default or explicit scale are represented without [dropout](../../../../glossary.md#dropout--inverted-dropout), hidden random state,
public attention [weights](../../../../glossary.md#term-weight), [gradients](../../../../glossary.md#term-gradient), [decomposition](../../../../glossary.md#term-decomposition), execution, or backend policy.

This is first-class rather than literal [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/mask/[softmax](../../../../glossary.md#softmax--log-softmax)/MATMUL composition so the selected
all-masked-row, masked-special-value, positive-[infinity](../../../../glossary.md#term-infinity)-tie, and stability contracts survive
[compiler](../../../../glossary.md#term-compiler) inspection. A future compiler may decompose it only while preserving the whole meaning;
backend [prepare](../../../../glossary.md#prepare) may choose a conforming [fused](../../../../glossary.md#specialization-and-fusion) or unfused implementation.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

```text
query  [..., L, E]
key    [..., S, E]  -> scores/weights [..., L, S]
value  [..., S, Ev] -> output         [..., L, Ev]

scores  = (query @ transpose(key)) * scale
weights = maskedSoftmax(scores, final axis S)
output  = weights @ value
```

Leading [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/head prefixes broadcast right-aligned across query, key, and value. A [mask](../../../../glossary.md#term-boolean-mask)
broadcasts exactly to the score [Shape](../../../../glossary.md#shape). `true` includes and `false` excludes a position before its
score or value special values participate. Causal mode additionally requires `j <= i` for
zero-based query position `i` and key position `j`, including rectangular `L`/`S`.

### Conceptual [metadata](../../../../glossary.md#term-metadata) example

The documentation pass must include a clearly labeled conceptual example equivalent to:

```java
Tensor output = query.scaledDotProductAttention(
        key,
        value,
        mask,
        new ScaledDotProductAttentionAttrs(Optional.empty(), true));
```

For `query=[2,4,8]`, `key=[1,6,8]`, `value=[2,6,10]`, and `mask=[4,6]`, the
score [Shape](../../../../glossary.md#shape) is `[2,4,6]` and output Shape is `[2,4,10]`. Scale is `1 / sqrt(8)`, [mask](../../../../glossary.md#term-boolean-mask) broadcasts
across [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), and causal mode also excludes `j > i`. This shows expression metadata, not numeric
execution, [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings), [gradients](../../../../glossary.md#term-gradient), or [backend](../../../../glossary.md#backend) support.

## [Scope](../../../../glossary.md#term-scope)

- Add one public `SCALED_DOT_PRODUCT_ATTENTION` kind in a focused attention package.
- Add public [immutable](../../../../glossary.md#term-immutable) `ScaledDotProductAttentionAttrs(Optional<ScalarValue> scale,
  boolean causal)`.
- Give the kind one three-to-four-input, exactly-one-output [signature](../../../../glossary.md#term-signature).
- Add exactly four unambiguous receiver methods: key/value defaults; key/value/attrs;
  key/value/mask defaults; and key/value/mask/attrs.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Add one package-private field-free helper owning all local validation, Shape inference,
  descriptor/operation construction, and one factory delegation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [inference glossary](../../../../glossary.md#term-inference); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation))</span>
- Define exact Shape, type, mask, scale, numerical, metadata, provenance, validation-order,
  freshness, side-effect, and deferred-constraint contracts.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Add semantic/expression tests and update every global public-[`Tensor`](../../../../glossary.md#tensor) inventory/count from 171
  to 175.
- Finalize Javadocs, Tensor/Compile API, glossary, capabilities, and planning in a mandatory
  separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- [dropout](../../../../glossary.md#dropout--inverted-dropout), training mode, [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), hidden/global RNG, random draws, dropout masks, or state
  output<span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask))</span>
- public attention [weights](../../../../glossary.md#term-weight), auxiliary/saved outputs, multi-output [carrier](../../../../glossary.md#term-storage-carrier), or saved statistics
- gradients, adjoints, backward operations, trainable parameters, compiler capture, graph-wide
  solving, canonicalization, decomposition, fusion, or optimization<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [trainable parameters glossary](../../../../glossary.md#trainable-parameter); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- algorithms, online softmax, tiling, materialization, fixed accumulation order, kernels, backend
  capability/lowering, prepare, runtime, execution, conformance, or integration<span data-glossary-navigation="true"> ([tiling glossary](../../../../glossary.md#tiling); [materialization glossary](../../../../glossary.md#materialization); [kernels glossary](../../../../glossary.md#kernel); [backend capability glossary](../../../../glossary.md#backend-capability); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- additive masks/bias, offsets, causal variants, grouped-query/sparse attention, caches, packed
  sequences, quantized/complex/unsigned types, or FLOAT16<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
- runtime scalar Tensor scale, nullable mask, null-as-default, NaN sentinel, cast insertion,
  primitive scale overloads, broad options framework, or <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Tensor glossary](../../../../glossary.md#tensor); [NaN glossary](../../../../glossary.md#term-nan))</span>`Optional<Tensor>`
- changes to [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), [softmax](../../../../glossary.md#softmax--log-softmax), where, masked reductions, dropout/[RNG](../../../../glossary.md#graph-rng-state--graphrngstate), [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), promotion,
  `ShapeBroadcast`, [Dimension](../../../../glossary.md#dimension), [provenance](../../../../glossary.md#provenance), [signatures](../../../../glossary.md#term-signature), or factory contracts
- Gradle, another module, architecture/ADR/test, focused architecture, backend-conformance, or
  integration changes<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend))</span>

## Exact public [API](../../../../glossary.md#term-application-programming-interface-api) and attributes

The receiver is query. Add exactly:

```java
public Tensor scaledDotProductAttention(Tensor key, Tensor value)
public Tensor scaledDotProductAttention(
        Tensor key, Tensor value, ScaledDotProductAttentionAttrs attrs)
public Tensor scaledDotProductAttention(Tensor key, Tensor value, Tensor mask)
public Tensor scaledDotProductAttention(
        Tensor key, Tensor value, Tensor mask, ScaledDotProductAttentionAttrs attrs)
```

The short forms use `new ScaledDotProductAttentionAttrs(Optional.empty(), false)`. Causal mode or
explicit scale uses attrs. [Tensor](../../../../glossary.md#tensor)-versus-attrs third [parameters](../../../../glossary.md#term-parameter) avoid ambiguity. Add no [boolean](../../../../glossary.md#data-type--datatype),
primitive-scale, nullable-[mask](../../../../glossary.md#term-boolean-mask), varargs, builder, or general-options overload.

The attrs record is public because public [provenance](../../../../glossary.md#provenance) exposes exact typed semantic attributes. It
is [operation](../../../../glossary.md#operation)-specific, not a broad options framework. `Optional.empty()` has one genuine meaning:
compute `1 / sqrt(E)` after [embedding](../../../../glossary.md#embedding) [extent](../../../../glossary.md#term-extent) [binding](../../../../glossary.md#term-binding). A present exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) is the scale.

The record constructor null-checks `scale`; when present it requires [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16),
decodes the exact value, and requires it [finite](../../../../glossary.md#term-finite) and strictly positive. Retain exact bits. Use:

```text
scale must have a floating data type, but was <dataType>
scale must be finite and positive: <value>
```

Zero and negative values erase/reverse similarity order and are not selected attention semantics;
[NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity) are invalid configuration rather than score data.

## [Operation](../../../../glossary.md#operation) [model](../../../../glossary.md#term-model)

Add public types under `io.github.pho001.synaptik.model.operation.attention`:

- `ScaledDotProductAttentionKind`, with exactly `SCALED_DOT_PRODUCT_ATTENTION`;
- `ScaledDotProductAttentionAttrs`, the record above.

The kind's exact stable [signature](../../../../glossary.md#term-signature) is:

```java
OperationSignature.inputRange(ScaledDotProductAttentionAttrs.class, 3, 4, 1)
```

Unmasked inputs are `[query,key,value]`; masked inputs are `[query,key,value,mask]`. Both create
one [descriptor](../../../../glossary.md#term-descriptor)/output, one wrapper, one [producer](../../../../glossary.md#term-producer), one ID, and [provenance](../../../../glossary.md#provenance) output index zero. There
is no attention-[weight](../../../../glossary.md#term-weight)/state output or [dropout](../../../../glossary.md#dropout--inverted-dropout) [parameter](../../../../glossary.md#term-parameter). The structural range does not replace
public/helper role [validation](../../../../glossary.md#term-validation) or later [compiler](../../../../glossary.md#term-compiler) validation.

## [Shape](../../../../glossary.md#shape) and deferred-constraint contract

### [Axes](../../../../glossary.md#term-axis) and ranks

Require query, key, and value ranks at least two:

```text
query = [...queryBatch, L, E]
key   = [...keyBatch,   S, E]
value = [...valueBatch, S, Ev]
```

Final two axes are [matrix](../../../../glossary.md#term-matrix-meanings) axes; no [vector](../../../../glossary.md#term-vector) promotion exists. Static-zero query `E` fails. Dynamic
`E` defers positivity. Unequal static query/key `E` fails; any unresolved pair defers equality
because `E` is absent from output. Unequal static key/value `S` fails; unresolved pairs defer
equality. Name these obligations `attention embedding positivity`, `attention embedding
equality`, and `attention key/value sequence equality`. No [model](../../../../glossary.md#term-model) constraint object is added;
future [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings) re-derives and proves them.

### Three-way [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/head [broadcasting](../../../../glossary.md#broadcasting)

Right-align all three prefixes together. Missing [axes](../../../../glossary.md#term-axis) behave as [singleton](../../../../glossary.md#term-singleton). Per result batch axis:

1. reject more than one distinct static non-singleton [extent](../../../../glossary.md#term-extent);
2. with one static non-singleton, retain its exact first reference in query/key/value order and
   defer each unequal unresolved participant to singleton-or-that-extent;
3. otherwise ignore static singletons; remaining unresolved Dimensions must be structurally equal
   or reject because no exact output is derivable;
4. retain the first remaining unresolved reference in query/key/value order, or if all singleton/
   missing retain the first present input singleton reference.

Unpaired axes retain exact references. Named/expression equality is structural; constrained
unknown equality follows current [identity](../../../../glossary.md#term-identity-meanings) semantics. Deferred obligations are `attention batch
broadcast` constraints. Do not change/use pairwise `ShapeBroadcast`, synthesize unknown output,
or bind [dimensions](../../../../glossary.md#dimension).

### Score, output, and [mask](../../../../glossary.md#term-boolean-mask)

```text
score  = [...exact broadcastBatch, exact query L, exact key S]
output = [...exact broadcastBatch, exact query L, exact value Ev]
```

New outer [Shapes](../../../../glossary.md#shape) retain all selected exact [Dimension](../../../../glossary.md#dimension) references.

A mask must be exact [BOOL](../../../../glossary.md#data-type--datatype) and [rank](../../../../glossary.md#term-rank) no greater than score rank. [Scalar](../../../../glossary.md#term-scalar) mask is valid: true admits
all otherwise eligible positions; false masks all. Accept every rank zero through score rank that
right-broadcasts exactly to the preselected score Shape. Per aligned [axis](../../../../glossary.md#term-axis):

- static mask [singleton](../../../../glossary.md#term-singleton) or structurally equal mask Dimension is accepted, retaining score ref;
- unresolved mask against static non-singleton score defers singleton-or-score equality;
- every other unequal pair rejects, including static non-singleton mask against unresolved score
  and two unequal unresolved Dimensions.

Unpaired leading score axes see implicit mask singletons. Deferred obligations are `attention mask
broadcast` constraints. Mask never changes score/output refs or uses `ShapeBroadcast`.

Causal eligibility is exactly `j <= i`, top-left aligned with no offset/window/right [alignment](../../../../glossary.md#term-alignment).
Explicit and causal masks combine by logical AND.

## Types, scale, and [metadata](../../../../glossary.md#term-metadata)

Require query, key, value floating in that order. Result type is:

```java
DataType queryKeyType = DataTypePromotion.promoteFloating(queryType, keyType);
DataType resultType = DataTypePromotion.promoteFloating(queryKeyType, valueType);
```

No casts. Explicit scale must exactly match `resultType`:

```text
scale data type must match promoted attention data type: scale=<scaleType>, promoted=<resultType>
```

Absent scale remains absent even for static `E` and means exact semantic `1 / sqrt(E)` after
positive [binding](../../../../glossary.md#term-binding); [model](../../../../glossary.md#term-model) does not precompute/round it. No [scalar](../../../../glossary.md#term-scalar) [Tensor](../../../../glossary.md#tensor) is needed because scale is
[occurrence](../../../../glossary.md#term-occurrence) configuration, not [graph](../../../../glossary.md#graph) data.

Result [descriptor](../../../../glossary.md#term-descriptor): promoted type; exact output [Shape](../../../../glossary.md#shape); unresolved [layout](../../../../glossary.md#layout); requires-grad OR of
query/key/value only; empty label/storage. [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) is non-[gradient](../../../../glossary.md#term-gradient). One successful call creates
fresh descriptor, [operation](../../../../glossary.md#operation), [producer](../../../../glossary.md#term-producer), wrapper, ID. Attrs-bearing overloads retain the exact attrs
reference; inputs are exact ordered refs. Repeated calls are [identity](../../../../glossary.md#term-identity-meanings)-distinct and inputs/attrs
remain unchanged.

## Numerical semantics

For each [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/query row, compute eligible ideal dot-product scores times scale, [softmax](../../../../glossary.md#softmax--log-softmax) only over
eligible final-[axis](../../../../glossary.md#term-axis) `S` positions, then weighted value sum. Excluded positions have positive-zero
[weight](../../../../glossary.md#term-weight) and are excluded before score and value arithmetic; masked key/value [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity) cannot
contaminate the row.

Per row:

- no eligible positions: all weights/output components [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero);
- any eligible NaN score: eligible weights and non-empty output components NaN, excluded weights
  positive zero;
- one or more eligible positive infinities: split unit weight equally among them and give all
  other eligible positions positive-zero weight; weighted value arithmetic still follows the
  general eligible-position rule below;
- all eligible scores [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero): weights/output positive zero, not NaN;
- otherwise finite scores use ideal exponential ratios, eligible negative infinities get zero,
  and ideal weights total one.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>

Thus explicit/causal all-masked rows and `S==0` rows are zero. Eligible query/key NaN follows dot
propagation. Eligible value NaN follows floating multiply/add even if its non-masked weight is zero
from a negative-infinity score; only eligibility exclusion guarantees ignoring it.

[FLOAT64](../../../../glossary.md#term-float64) accumulates scores and output sums in FLOAT64. [FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16) accumulate both in
FLOAT32; BFLOAT16 converts the final output. Scale participates in that domain. [Reassociation](../../../../glossary.md#term-reassociation),
[fused multiply-add](../../../../glossary.md#term-fused-multiply-add-fma), and any stable softmax satisfying special rules are allowed. Later [conformance](../../../../glossary.md#term-conformance)
sets tolerances. No traversal order, bitwise result, cross-[backend](../../../../glossary.md#backend) identical [rounding](../../../../glossary.md#term-rounding), or algorithm
is promised. Ordinary [IEEE](../../../../glossary.md#term-ieee) multiplication/addition governs eligible [infinities](../../../../glossary.md#nan-infinity-and-signed-zero)/[signed zero](../../../../glossary.md#term-signed-zero) subject
to reassociation; only mandated empty/all-masked/all-negative-infinity zero is positive zero.

Empty decisions: `L==0` is valid with no rows; `S==0` is valid with positive-zero output for every
existing query row; static `E==0` is invalid and dynamic `E` defers positivity; `Ev==0` is valid
with an empty output axis; zero batch/head axes are valid. Determinism fixes mathematical
eligibility/ties/results for one conforming prepared implementation, not bitwise portability.

## [Validation](../../../../glossary.md#term-validation) order, failures, and ID effects

Complete every local check before [descriptor](../../../../glossary.md#term-descriptor)/[operation](../../../../glossary.md#operation)/[producer](../../../../glossary.md#term-producer)/ID creation:

1. null-check query, key, value, [mask](../../../../glossary.md#term-boolean-mask) when present, attrs;
2. validate query/key/value floating and promote in order;
3. validate query/key/value ranks in order;
4. reject static-zero query [embedding](../../../../glossary.md#embedding);
5. reject static query/key embedding mismatch;
6. reject static key/value sequence mismatch;
7. derive [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [axes](../../../../glossary.md#term-axis) leading to trailing;
8. build local score/output [Shapes](../../../../glossary.md#shape);
9. validate present mask [BOOL](../../../../glossary.md#data-type--datatype), [rank](../../../../glossary.md#term-rank), then axes leading to trailing;
10. validate present scale exact promoted type (attrs already validates floating/[finite](../../../../glossary.md#term-finite)/positive);
11. construct descriptor/operation;
12. delegate once to `TensorFactory.createDerived` with exact inputs.

Use exact messages:

```text
query must have a floating data type, but was <dataType>
key must have a floating data type, but was <dataType>
value must have a floating data type, but was <dataType>
query rank must be at least 2: <rank>
key rank must be at least 2: <rank>
value rank must be at least 2: <rank>
attention embedding dimension must be positive: <dimension>
attention query/key embedding dimensions must match: query=<dimension>, key=<dimension>
attention key/value sequence dimensions must match: key=<dimension>, value=<dimension>
cannot broadcast attention batch dimensions at result batch axis <axis>: query=<dimension>, key=<dimension>, value=<dimension>
cannot derive exact attention batch dimension at result batch axis <axis>: query=<dimension>, key=<dimension>, value=<dimension>
mask must have BOOL data type, but was <dataType>
mask rank must not exceed attention score rank: mask=<maskRank>, score=<scoreRank>
mask cannot broadcast exactly to attention score shape at axis <axis>: mask=<dimension>, score=<dimension>
scale data type must match promoted attention data type: scale=<scaleType>, promoted=<resultType>
```

The first batch message is conflicting static non-singletons; the second is unequal unresolved
candidates without a selecting static [extent](../../../../glossary.md#term-extent). Embedding/sequence mismatch messages apply only to
unequal statics. Every failure through operation construction consumes no ID/producer/wrapper;
successful calls consume exactly one ID. Factory exhaustion retains current no-[rollback](../../../../glossary.md#term-rollback) behavior.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Signature hardening](0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Signature glossary](../../../../glossary.md#term-signature))</span>
- [Shared provenance](0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- [Typed scalar](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Mask redesign](0018q-masked-reduction-redesign.md)<span data-glossary-navigation="true"> ([Mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- [MATMUL](0019-matmul-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- [Softmax semantics](0016i-softmax-semantic-kinds-and-attributes.md)<span data-glossary-navigation="true"> ([Softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- [Softmax expressions](0016j-softmax-tensor-expressions.md)

## Architecture constraints

- Work stays in [model](../../../../glossary.md#term-model) plus its documentation/[planning](../../../../glossary.md#term-planning). [Tensor](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) state, not [IR](../../../../glossary.md#term-intermediate-representation-ir).
- Attention types record [backend](../../../../glossary.md#backend)-independent meaning and no support/route/storage/[runtime](../../../../glossary.md#term-runtime) behavior.
- Direction is tensor -> attention operation/datatype/shape; operation attention imports no Tensor,
  graph, compiler, runtime, prepare, or backend type.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [shape glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph); [prepare glossary](../../../../glossary.md#prepare))</span>
- [Compiler](../../../../glossary.md#term-compiler) owns [capture](../../../../glossary.md#term-capture-meanings), deferred proofs, legal [decomposition](../../../../glossary.md#term-decomposition), [gradients](../../../../glossary.md#term-gradient), adjoints/saved values.
- Backend prepare owns conforming fused/unfused lowering, algorithms, materialization, kernels;
  runtime executes prepared work without original operations on its hot path.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion); [lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization); [kernels glossary](../../../../glossary.md#kernel); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- Attention has no technical [dropout](../../../../glossary.md#dropout--inverted-dropout) [dependency](../../../../glossary.md#term-dependency). Future attention dropout must consume explicit
  [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), never hidden/global [RNG](../../../../glossary.md#graph-rng-state--graphrngstate).
- No architecture, dependency, [lifecycle](../../../../glossary.md#lifecycle), focused-architecture, [Gradle](../../../../glossary.md#term-gradle), or cross-[module](../../../../glossary.md#term-module) change.

## Package impact

Existing: datatype, [shape](../../../../glossary.md#shape), [operation](../../../../glossary.md#operation), [tensor](../../../../glossary.md#tensor). Add
`io.github.pho001.synaptik.model.operation.attention` for public attention [identity](../../../../glossary.md#term-identity-meanings) and attrs only.

Type placement:

- `...attention.ScaledDotProductAttentionKind` — identity/[signature](../../../../glossary.md#term-signature) [owner](../../../../glossary.md#term-owner).
- `...attention.ScaledDotProductAttentionAttrs` — public inspectable [immutable](../../../../glossary.md#term-immutable) semantics.
- `...tensor.TensorScaledDotProductAttentionExpressions` — package-private construction owner.
- `...tensor.Tensor` — established public fluent [facade](../../../../glossary.md#term-facade).

Tests mirror production packages for package-private inspection.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (6):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — four exact
  signatures and count 171 -> 175.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java` —
  count only, 171 -> 175.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java` —
  count only, 171 -> 175.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java` —
  count only, 171 -> 175.

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime)/Training APIs; related [operation](../../../../glossary.md#operation)/type/[shape](../../../../glossary.md#shape)/[provenance](../../../../glossary.md#provenance)
contracts; architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 17 paths maximum: four production, six test, seven documentation/[planning](../../../../glossary.md#term-planning). `Tensor.java`
changes only imports, four methods/Javadocs, class [operation](../../../../glossary.md#operation) inventory. Four existing tests change
only stated global inventories/counts. Stop for path 18, another type/test, existing helper change,
architecture, [Gradle](../../../../glossary.md#term-gradle), or cross-[module](../../../../glossary.md#term-module) work.

This is cohesive and inside the planning guide's 12–18 path guardrail. Kind, attrs, public
construction, [API](../../../../glossary.md#term-application-programming-interface-api) locks, and docs must agree in one compilable capability; splitting creates an
unusable intermediate and duplicates one tightly coupled numerical/[mask](../../../../glossary.md#term-boolean-mask) review. No 0019E1 split.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document kind, attrs, helper, and four Tensor methods: API, Shape/type/mask/scale/causal,
  numerical/empty, metadata/provenance, validation/failures, freshness, and layer boundaries.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Shape glossary](../../../../glossary.md#shape); [mask glossary](../../../../glossary.md#term-boolean-mask); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation))</span>
- Every [parameter](../../../../glossary.md#term-parameter)/return/failure has `@param`/`@return`/`@throws` as applicable.
- Tensor API gets mental model, overload/Shape tables, conceptual metadata and small eligibility
  examples, special/empty rules, and current-model versus planned execution boundary.<span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span><span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Compile API says current expression metadata and future compiler-owned capture/constraints/
  decomposition/gradients; it does not claim compiler support.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [decomposition glossary](../../../../glossary.md#term-decomposition); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Review glossary terms [scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention), [attention score](../../../../glossary.md#attention-score), [causal mask](../../../../glossary.md#causal-mask), [all-masked row](../../../../glossary.md#all-masked-row).
- Synchronize capabilities/task/master/roadmap: 0019–0019E Complete, no model task Ready, and
  0020+ Draft without detailed specs.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Record reasoned no-change conclusions for Runtime/Training API, related contracts, architecture,
  conformance/integration, Gradle, and other modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Exact kind/attrs/input range/one output and exact four receiver overloads exist; count is 175.
- All [rank](../../../../glossary.md#term-rank)/[axis](../../../../glossary.md#term-axis)/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[embedding](../../../../glossary.md#embedding)/sequence/score/output/[Dimension](../../../../glossary.md#dimension)-reference rules pass.
- [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) ranks/[scalar](../../../../glossary.md#term-scalar)/dynamic contracts and exact fourth input pass.
- Default dynamic scale and exact typed [finite](../../../../glossary.md#term-finite)-positive explicit scale pass; no casts/scalar input.
- Promotion/accumulation/[gradient](../../../../glossary.md#term-gradient)/[layout](../../../../glossary.md#layout)/label/storage/freshness/[provenance](../../../../glossary.md#provenance)/ID contracts pass.
- Masked special values, eligible NaN, +infinity ties, all -infinity/all-masked rows, signed-zero,
  empty extents, reassociation/tolerance/determinism are documented/tested without evaluation.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [reassociation glossary](../../../../glossary.md#term-reassociation); [tolerance glossary](../../../../glossary.md#term-tolerance); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Exact [validation](../../../../glossary.md#term-validation) order/messages and no-ID failures pass; success allocates one fresh ID.
- No [dropout](../../../../glossary.md#dropout--inverted-dropout)/[weights](../../../../glossary.md#term-weight)/multi-output/gradient/[compiler](../../../../glossary.md#term-compiler)/algorithm/[backend](../../../../glossary.md#backend)/[runtime](../../../../glossary.md#term-runtime)/architecture work.
- Focused tests, exactly one final model suite, Javadoc/docs/link/anchor/fence/newline/whitespace,
  exact 17 paths, packages, public surface, and statuses pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Separate clean documentation pass reuses Java evidence and records no-change conclusions.
- 0019–0019E are Complete; no [model](../../../../glossary.md#term-model) task is [Ready](../../../../glossary.md#term-ready); 0020+ remain Draft without specs.

## Tests / [validation](../../../../glossary.md#term-validation)

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest --tests io.github.pho001.synaptik.model.datatype.DataTypePromotionTest --tests io.github.pho001.synaptik.model.shape.ShapeBroadcastTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Focused tests cover semantic/helper/public surfaces, overloads, all static/dynamic [Shape](../../../../glossary.md#shape) cases,
floating combinations/scales, [mask](../../../../glossary.md#term-boolean-mask) ranks/[scalar](../../../../glossary.md#term-scalar), causal rectangles, validation/ID effects,
[metadata](../../../../glossary.md#term-metadata)/[provenance](../../../../glossary.md#provenance)/freshness, and numerical contracts as metadata/[Javadoc](../../../../glossary.md#term-javadoc) without value execution.

Documentation pass after final Javadoc:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

Also check local Markdown links/anchors, fences, terminology, conceptual labels, generated Javadoc,
newlines/whitespace, exact paths/packages, public count/[signatures](../../../../glossary.md#term-signature), exactly one [Ready](../../../../glossary.md#term-ready) frontier,
0019–0019D Complete, 0020+ Draft, and no later specs. Repository validation is deferred to the
selected-modern-operations [checkpoint](../../../../glossary.md#term-checkpoint) after 0022 or [CI](../../../../glossary.md#term-continuous-integration-ci); no repository-wide contract changes.

## Dependencies

- 0001–0002, 0018M–0018M1: types, [Dimensions](../../../../glossary.md#dimension), [Shapes](../../../../glossary.md#shape), symbolic extents.
- 0005–0007, 0011–0013, 0018K–0018L: operations/[signatures](../../../../glossary.md#term-signature), [Tensor](../../../../glossary.md#tensor)/factory/[provenance](../../../../glossary.md#provenance).
- 0016I–0016J: [softmax](../../../../glossary.md#softmax--log-softmax) semantics/terminology.
- 0018N: exact typed [scalar](../../../../glossary.md#term-scalar).
- 0018Q: [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) exclude-before-value precedent.
- 0019: [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [axes](../../../../glossary.md#term-axis)/promotion/accumulation/deferred-[contraction](../../../../glossary.md#contraction-dimension) precedent.

0019B/0019B1 is table-order history, not a technical [dependency](../../../../glossary.md#term-dependency).

## Follow-up tasks

- 0020 stays Draft/unchanged.
- [Compiler](../../../../glossary.md#term-compiler) later owns [capture](../../../../glossary.md#term-capture-meanings)/constraints/[decomposition](../../../../glossary.md#term-decomposition)/[gradients](../../../../glossary.md#term-gradient)/adjoints/saved values.
- Backend/conformance/runtime/integration later own algorithms/tolerances/lowering/kernels/storage/
  prepared execution/numeric evidence.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [runtime glossary](../../../../glossary.md#term-runtime); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- Future attention [dropout](../../../../glossary.md#dropout--inverted-dropout) is separate and must consume explicit [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate).

Do not create another detailed spec during implementation.

## Architecture impact

Expected impact: None. Stop and report any architecture, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), focused-doc,
cross-[module](../../../../glossary.md#term-module), or [scope](../../../../glossary.md#term-scope) need.

## Implementation prompt

```text
Work in Synaptik without commit or push. Read AGENTS.md, ARCHITECTURE.md, focused architecture,
documentation/planning rules, roadmap, model capabilities/master, completed operation/signature/
Tensor/provenance/shared-output/shape/broadcast/typed-scalar/MATMUL/softmax/mask/dropout/linear
tasks, current related source/tests/APIs/glossary, and task 0019E.

Implement task 0019E exactly inside 17 paths. Update every global public Tensor inventory/count
171 -> 175 up front. Preserve contracts. Stop on architecture uncertainty, scope overflow,
another type/test/document, existing-helper change, or cross-module work.

Run focused validation and exactly one final model suite after Java stabilizes. Hand actual diff
and Java evidence to a separate clean documentation agent in the same change; it finalizes
Javadocs, Tensor/Compile APIs, glossary, capabilities/planning and docs checks while reusing Java
evidence. Complete only after all criteria; keep 0019–0019D Complete and 0020+ Draft/no specs.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Provide task, complete diff, exact focused/final evidence and post-test Java-change state,
[API](../../../../glossary.md#term-application-programming-interface-api)/[Shape](../../../../glossary.md#shape)/type/[mask](../../../../glossary.md#term-boolean-mask)/scale/numerical/[provenance](../../../../glossary.md#provenance) policies, seven docs paths, and [validation](../../../../glossary.md#term-validation). The clean
agent reads AGENTS/architecture/rules/profiles, task, source/tests/generated [Javadoc](../../../../glossary.md#term-javadoc), all four APIs,
glossary/[planning](../../../../glossary.md#term-planning), and related contracts; finalizes docs and records reasoned no-change conclusions.
It does not repeat successful Java tests absent executable change/stale evidence/concrete risk.

## Local decisions

- Kept scaled dot-product attention first-class so eligibility and special-value rules remain
  inspectable rather than becoming an implicit primitive chain.<span data-glossary-navigation="true"> ([scaled dot-product attention glossary](../../../../glossary.md#scaled-dot-product-attention))</span>
- Used one public operation-specific attrs record and exactly four overloads; absent scale remains
  semantic state rather than a precomputed static value.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Kept every unresolved equality, positivity, and singleton-or-equal fact deferred without adding
  a model constraint object or changing shared broadcasting.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [model glossary](../../../../glossary.md#term-model); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>

## Known limitations

- Deferred attention constraints are recorded semantically, not solved in [model](../../../../glossary.md#term-model).
- [FLOAT16](../../../../glossary.md#term-float16), [dropout](../../../../glossary.md#dropout--inverted-dropout), [bias](../../../../glossary.md#term-bias)/additive [masks](../../../../glossary.md#term-boolean-mask), [weights](../../../../glossary.md#term-weight), [gradients](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler)/[backend](../../../../glossary.md#backend)/execution are absent.
- [Reassociation](../../../../glossary.md#term-reassociation)/[stable algorithm](../../../../glossary.md#conditioning-and-numerical-stability) freedom means no cross-backend bitwise promise.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context: `/root/task_0019e_implementation`.

- The prescribed seven-suite focused command passed 43 tests with zero failures, errors, or skips.
- After executable Java stabilized, exactly one actual final `./gradlew :modules:model:test` run
  passed 848 tests across 107 XML suites with zero failures, errors, or skips. An earlier attempted
  chained final command did not reach Gradle because wrapper-cache lock access was denied; it is
  not counted as a test run. No executable Java changed after the successful final suite.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [Gradle glossary](../../../../glossary.md#term-gradle); [cache glossary](../../../../glossary.md#term-cache))</span>
- `git diff --check` passed after Java stabilization. The separate documentation context changed
  only Javadocs and the seven authorized documentation/planning paths, so it did not repeat the
  successful Java suites.

Documentation context:
`/root/task_0019e_implementation/task_0019e_docs`, applying the General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Planning](../../../../glossary.md#term-planning),
and Example profiles.

- Independently reviewed repository instructions; the architecture and focused module boundary;
  documentation/planning rules; roadmap, capabilities, master plan, and this task; all four
  production and six test paths; Tensor/Compile/Runtime/Training APIs; glossary; and directly
  related operation, signature, Tensor, provenance, shared-output, Shape/broadcast, typed-scalar,
  MATMUL, softmax, mask, dropout, and linear contracts.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [provenance glossary](../../../../glossary.md#provenance); [Shape glossary](../../../../glossary.md#shape); [scalar glossary](../../../../glossary.md#term-scalar); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- Finalized Javadocs for the attention kind, attrs, package-private helper, Tensor class inventory,
  and all four public receiver methods. Parameter, return, failure, Shape/type/mask/scale/causal,
  numerical, provenance, freshness, side-effect, and lifecycle boundaries agree with source and
  focused tests. No executable statement changed.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Tensor API now includes the Shape/overload mental model, eligibility and numerical rules, and
  conceptual <span data-glossary-navigation="true"> ([mental model glossary](../../../../glossary.md#term-mental-model))</span>`[2,4,8]` query, `[1,6,8]` key, `[2,6,10]` value, `[4,6]` [mask](../../../../glossary.md#term-boolean-mask) example deriving score
  `[2,4,6]` and output `[2,4,10]`. Compile API records current model metadata and future
  compiler-owned proof, decomposition, and gradient work without claiming compiler support.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span><span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [decomposition glossary](../../../../glossary.md#term-decomposition); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Glossary defines scaled dot-product attention, attention score, causal mask, and all-masked row.
  Capabilities, master plan, roadmap, and this task consistently mark 0019 through 0019E Complete;
  task 0020 and later remain Draft without detailed specifications and no model task is Ready.<span data-glossary-navigation="true"> ([scaled dot-product attention glossary](../../../../glossary.md#scaled-dot-product-attention); [attention score glossary](../../../../glossary.md#attention-score); [causal mask glossary](../../../../glossary.md#causal-mask); [all-masked row glossary](../../../../glossary.md#all-masked-row); [Ready glossary](../../../../glossary.md#term-ready))</span>
- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL`; two tasks executed. Generated
  Javadoc contains the public attrs/kind pages and exactly four public Tensor attention overloads.
  The package-private helper is absent from the public generated surface.
- `javap` confirmed 175 public Tensor methods excluding the constructor and exactly the four
  required overload descriptors. Package/import scans found the two public attention types in the
  focused operation package, the helper in the Tensor package, and no compiler, runtime, prepare,
  backend, training, or cross-module import.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- Targeted Markdown validation resolved 575 local links and 156 heading fragments across the seven
  changed Markdown files. Fences are balanced; every changed/new path is non-empty, ends with a
  newline, and has no trailing whitespace.
- Exact-scope validation found only the authorized 17 paths: four production, six tests, and seven
  documentation/planning paths. No task 0020-or-later specification exists. Final status,
  terminology, package, public-surface, generated-Javadoc, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` audits passed.
- Runtime API needs no change because this task creates no prepared execution, schedule, resource,
  binding, run-state, residency, or executable contract. Training API needs no change because it
  adds no parameter ownership, dropout state, gradient formula, backward operation, optimizer, or
  training workflow.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [binding glossary](../../../../glossary.md#term-binding); [residency glossary](../../../../glossary.md#residency); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Related operation/signature, Tensor/provenance/shared-output, Shape/broadcast, typed-scalar,
  MATMUL, softmax, mask, dropout, and linear contracts remain accurate because attention adds a
  separate focused family without changing them. Architecture/ADRs/tests, conformance/integration,
  Gradle, and other modules need no change because module ownership, dependency direction, build
  structure, and executable backend/runtime behavior are unchanged.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>

## Implementation notes

Implemented [immutable](../../../../glossary.md#term-immutable) attention attrs and one exact three-to-four-input/one-output kind, then
delegated the four public overloads to one package-private stateless helper. The helper completes
all local [validation](../../../../glossary.md#term-validation) and exact [Shape](../../../../glossary.md#shape) selection before one derived-[Tensor factory](../../../../glossary.md#tensor-factory) call. The clean
documentation pass finalized Javadocs and the seven authorized documentation/[planning](../../../../glossary.md#term-planning) paths
without changing executable Java.

## Completion summary

- Completed changes: added first-class one-output scaled dot-product attention semantics, immutable
  scale/causal attrs, exact Shape/type/mask/deferred-constraint policy, and four public receiver
  overloads without dropout or attention-weight output.<span data-glossary-navigation="true"> ([scaled dot-product attention glossary](../../../../glossary.md#scaled-dot-product-attention); [immutable glossary](../../../../glossary.md#term-immutable); [Shape glossary](../../../../glossary.md#shape); [mask glossary](../../../../glossary.md#term-boolean-mask); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [weight glossary](../../../../glossary.md#term-weight))</span>
- Files changed or created: exactly four production Java, six model-test, and seven
  documentation/planning paths authorized by this task.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the passing 43-test focused selection and the sole successful final
  848-test/107-suite model run; model Javadoc, generated-page, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`, import/package, Markdown
  link/anchor/fence, terminology, exact-scope, status/no-later-spec, newline/whitespace, and<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` validations passed.
- Documentation-agent review: completed independently in the clean context named above.
- Documentation impact: Tensor and Compile APIs, glossary, capabilities, task, master plan, and
  roadmap now describe current attention metadata and future lifecycle boundaries. Runtime and
  Training APIs were reviewed and remain accurate without changes.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: all four affected production paths were reviewed and finalized; executable Java
  did not change after the recorded model suite.
- Glossary impact: added focused definitions for scaled dot-product attention, attention score,
  causal mask, and all-masked row.<span data-glossary-navigation="true"> ([attention score glossary](../../../../glossary.md#attention-score); [causal mask glossary](../../../../glossary.md#causal-mask); [all-masked row glossary](../../../../glossary.md#all-masked-row))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0019E. Task 0020 remains Draft.

Status: Complete
