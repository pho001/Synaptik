# Task 0022A: Dense-[Target](../../../../glossary.md#term-target) Categorical Cross-Entropy with Logits

**Editorial reading key (current context, not a status update).**

This record concerns the historical dense-target addition, not today’s full API or execution
coverage. The caller uses logits (unnormalized class scores) as receiver and supplies an exact-shape
floating target, class axis, and reduction. Model construction records ordered inputs `[logits,
target]`, normalized-axis metadata, and one storage-free loss producer; descriptions cross this
boundary, not evaluated values. It neither executes nor owns a result buffer. Dense weights are
caller-obligated to be finite, non-negative, and normalized; construction does not inspect or
renormalize them.

A group is one class slice: three class scores with target `[0, 0, 1]` measure disagreement with the
last class. Dense `MEAN` divides by the number of groups, not classes. Zero weights contribute
positive zero without erasing the specified NaN-logit boundaries. Follow the [current
categorical-loss
reference](../../../../reference/tensor/statistics-losses.md#categorical-cross-entropy-with-logits-expressions)
to distinguish later index/ignore forms, compiler responsibilities, and qualified execution routes
from this task’s exclusions. Model semantics defines meaning; execution providers realize supported
routes, not redefine it. Semantic responsibility does not imply storage ownership.

**Mathematical reading illustration.** For one group with $C\ge1$ classes, finite real logits $z_j$
and non-negative target weights $t_j$ summing to one, define $q_j=\exp(z_j)/\sum_{r=1}^{C}\exp(z_r)$
and loss $\ell=-\sum_{j=1}^{C}t_j\log q_j$, where $j,r\in\{1,\ldots,C\}$. For $C=2$, $z=(0,0)$ and
$t=(0,1)$, $\ell=\log 2$. These real-number probabilities explain the target meaning; they do not
authorize a separate rounded softmax-then-log implementation or alter exceptional-value rules.

## Status

Complete

## Goal

Add one intentionally narrow [backend](../../../../glossary.md#backend)-independent dense-[target](../../../../glossary.md#term-target) categorical cross-entropy meaning
that consumes logits and an exact-[shape](../../../../glossary.md#shape) floating target, normalizes one explicit [class axis](../../../../glossary.md#class-axis), and
applies the existing explicit `NONE`, `SUM`, or [`MEAN`](../../../../glossary.md#term-mean) [loss reduction](../../../../glossary.md#loss-reduction).

[Mental model](../../../../glossary.md#term-mental-model):

```text
logits + exact-shape dense probability target + class axis + explicit reduction
  -> one loss producer
     -> one weighted stable-log-softmax loss per non-class coordinate
        -> non-class Shape or one reduced scalar
```

The [operation](../../../../glossary.md#operation) remains one inspectable request. [Model](../../../../glossary.md#term-model) construction must not decompose it into
public [log-softmax](../../../../glossary.md#softmax--log-softmax), multiplication, or reduction expressions.

## [Scope](../../../../glossary.md#term-scope)

- Append exactly `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` to `LossKind` after
  `MEAN_SQUARED_ERROR`.
- Add [immutable](../../../../glossary.md#term-immutable) public
  `DenseCategoricalCrossEntropyWithLogitsAttrs(int axis, LossReduction reduction)`.
- Give the new kind exactly one fixed two-input/one-output signature while preserving the
  mean-squared-error kind and signature unchanged.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [mean glossary](../../../../glossary.md#term-mean))</span>
- Add exactly this fluent logits-receiver method:

  ```java
  public Tensor categoricalCrossEntropyWithLogits(
          Tensor target, int classAxis, LossReduction reduction)
  ```

- Extend the existing package-private, final, field-free `TensorLossExpressions` helper with one
  dense-target entry and only the narrowly shared private validation/Shape helpers justified by
  both loss constructions.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [loss glossary](../../../../glossary.md#loss))</span>
- Fix ordered roles, exact Shape compatibility, normalized class-axis metadata, result Shape,
  floating promotion, stable logits formula, dense-target value obligation, denominator, empty
  domains, special values, provenance, identifiers, diagnostics, and validation order.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Add focused semantic and Tensor-expression coverage and synchronize every existing exact public
  Tensor method-count inventory from 186 to 187.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Finalize affected Javadocs, Tensor and Compile API references, glossary impact, capabilities,
  task/master/roadmap records, and documentation checks through the mandatory clean
  documentation-focused handoff.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [handoff glossary](../../../../glossary.md#term-handoff))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- index targets, integer labels, class-axis-removed target input, ignore index, bounds checking,
  optional log-probability output, or any task-0022B implementation or detailed specification<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [axis glossary](../../../../glossary.md#term-axis); [ignore index glossary](../../../../glossary.md#ignore-index); [log glossary](../../../../glossary.md#term-log))</span>
- class weights, sample weights, masks, label smoothing, temperature, scale, epsilon, probability
  clipping, target renormalization, target broadcasting, implicit expansion, implicit casts,
  probability-input cross entropy, standalone negative-log-likelihood, binary cross entropy,
  Kullback–Leibler divergence, or another loss<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [masks glossary](../../../../glossary.md#term-boolean-mask); [epsilon glossary](../../../../glossary.md#epsilon); [broadcasting glossary](../../../../glossary.md#broadcasting); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [loss glossary](../../../../glossary.md#loss))</span>
- a default axis, default reduction, trailing-axis convenience, overload, static facade, alias,
  options object, parser, registry, primitive/string reduction spelling, or eager constant<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [alias glossary](../../../../glossary.md#aliasing-and-arena); [registry glossary](../../../../glossary.md#term-registry); [eager glossary](../../../../glossary.md#term-eager-evaluation); [constant glossary](../../../../glossary.md#term-constant))</span>
- value/storage reads during model construction, eager target validation, tolerances, a fixed
  max/sum traversal or reduction tree, public-primitive decomposition, or a backend algorithm<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation); [decomposition glossary](../../../../glossary.md#term-decomposition); [backend glossary](../../../../glossary.md#backend))</span>
- gradients, adjoints, autograd traversal, compiler capture/validation/decomposition/fusion,
  backend support/lowering/kernels, prepare, runtime, publication, execution, or training-session,
  optimizer, parameter, module, step, and checkpoint behavior<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [fusion glossary](../../../../glossary.md#term-fusion); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication); [optimizer glossary](../../../../glossary.md#term-optimizer); [parameter glossary](../../../../glossary.md#term-parameter); [module glossary](../../../../glossary.md#term-module); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [FLOAT16](../../../../glossary.md#term-float16), integral, [BOOL](../../../../glossary.md#data-type--datatype), complex, sparse, quantized, unsigned, or nullable operands/results
- changes to `LossReduction`, softmax/log-softmax, log-sum-exp, aggregate reductions, Shape,
  floating promotion, producer/provenance, factory, compiler/runtime/training Java APIs or
  behavior, architecture, dependencies, Gradle, another module, task 0022, or tasks 0023–0024<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-sum-exp glossary](../../../../glossary.md#log-sum-exp); [Shape glossary](../../../../glossary.md#shape); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Public and [operation](../../../../glossary.md#operation) contracts

### Kind, typed attributes, and [signature](../../../../glossary.md#term-signature)

`LossKind` contains exactly these values in order:

```java
MEAN_SQUARED_ERROR,
DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
```

The new kind's exact ordered signature is:

```java
OperationSignature.fixed(DenseCategoricalCrossEntropyWithLogitsAttrs.class, 2, 1)
```

One [occurrence](../../../../glossary.md#term-occurrence) has ordered inputs `[logits, target]` and ordered output `[loss]`. The receiver is
the logits input at position zero. There is no hidden [log](../../../../glossary.md#term-log)-probability input/output, saved value,
sibling result, or variable [cardinality](../../../../glossary.md#term-cardinality). `MEAN_SQUARED_ERROR` continues to accept only
`MeanSquaredErrorAttrs` with its current fixed 2/1 signature.

`DenseCategoricalCrossEntropyWithLogitsAttrs` has exactly these components and order:

```java
int axis,
LossReduction reduction
```

The stored [axis](../../../../glossary.md#term-axis) is already normalized to a non-negative logits axis. Its compact constructor:

1. rejects a negative axis with
   `IllegalArgumentException("axis must be non-negative: " + axis)`; then
2. null-checks reduction with `NullPointerException("reduction")`.

It retains the exact enum reference and stores no [Tensor](../../../../glossary.md#tensor), [Shape](../../../../glossary.md#shape), class [extent](../../../../glossary.md#term-extent), [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) count,
denominator, [target](../../../../glossary.md#term-target)-[normalization](../../../../glossary.md#term-normalization) flag, value, [data type](../../../../glossary.md#data-type--datatype), algorithm, [gradient](../../../../glossary.md#term-gradient), [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler),
[backend](../../../../glossary.md#backend), [runtime](../../../../glossary.md#term-runtime), or training state. It reuses `LossReduction` as a configuration component and
does not change that enum's members or meaning.

### Public receiver, ranks, [class axis](../../../../glossary.md#class-axis), and [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape)

The public surface is exactly:

```java
public Tensor categoricalCrossEntropyWithLogits(
        Tensor target, int classAxis, LossReduction reduction)
```

The receiver is logits. Logits must have [rank](../../../../glossary.md#term-rank) at least one because a [scalar](../../../../glossary.md#term-scalar) has no class axis.
`classAxis` accepts the current Shape range `[-rank, rank - 1]`, is normalized exactly once with
`logitsShape.normalizeAxis(classAxis)`, and only the normalized non-negative value is stored.

Target rank and every corresponding [Dimension](../../../../glossary.md#dimension) must match logits positionally:

- structurally equal Dimensions pass;
- unequal static Dimensions fail locally;
- a non-structurally-equal pair involving an unresolved Dimension is accepted with an equality
  obligation for later compiler capture/binding; and<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>
- no scalar, [singleton](../../../../glossary.md#term-singleton), leading-[axis](../../../../glossary.md#term-axis), right-aligned, or other [broadcasting](../../../../glossary.md#broadcasting) is accepted.

Result Shape is derived only from logits:

- `NONE` removes the normalized class axis and retains every other logits Dimension by exact
  reference and in order; removing the only axis returns canonical `Shape.scalar()`; and
- `SUM` and [`MEAN`](../../../../glossary.md#term-mean) return [canonical](../../../../glossary.md#term-canonical-form) `Shape.scalar()`.

The target Shape object is never substituted into the result. Exact Shape failures are:

```text
categoricalCrossEntropyWithLogits target rank must equal logits rank: logits=<logitsRank>, target=<targetRank>
categoricalCrossEntropyWithLogits target dimension mismatch at axis <axis>: logits=<logitsDimension>, target=<targetDimension>
```

An invalid raw axis retains the current Shape failure type and exact message:

```text
Axis <classAxis> is outside shape rank <logitsRank>
```

### Stable logits formula and dense-[target](../../../../glossary.md#term-target) obligation

For each non-class [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `g`, let `z[g,c]` be the logits [slice](../../../../glossary.md#slice), `t[g,c]` be the supplied
[dense target](../../../../glossary.md#dense-target) [weight](../../../../glossary.md#term-weight), and `C` be the eventual class [extent](../../../../glossary.md#term-extent). In the selected computation format:

```text
m[g]   = max_c(z[g,c])
lse[g] = m[g] + log(sum_c(exp(z[g,c] - m[g])))
q[g,c] = lse[g] - z[g,c]
loss[g] = sum_c(weightedContribution(t[g,c], q[g,c]))

weightedContribution(0, q) = +0
weightedContribution(t, q) = t * q, for t > 0
```

This is the stable-logits form of `-sum_c(t[g,c] * logSoftmax(z[g,:])[c])`. It must not first
materialize probabilities, take their logarithm, or expand into public [Tensor](../../../../glossary.md#tensor) primitives. A later
[backend](../../../../glossary.md#backend) may use any conforming [fused](../../../../glossary.md#specialization-and-fusion) or decomposed [stable algorithm](../../../../glossary.md#conditioning-and-numerical-stability), but naive `exp(z)` followed
by [normalization](../../../../glossary.md#term-normalization) is not a conforming [finite](../../../../glossary.md#term-finite)-[logit](../../../../glossary.md#logit) implementation when it overflows and changes
the selected result class. The explicit zero-weight convention gives an absent class exactly
positive-zero contribution even when its [log](../../../../glossary.md#term-log) probability is [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero).

For every evaluated group, callers have the value obligation that target entries are finite and
non-negative and describe a normalized [probability distribution](../../../../glossary.md#term-probability-distribution) along the [class axis](../../../../glossary.md#class-axis). [Model](../../../../glossary.md#term-model)
construction deliberately reads no target values, rejects no target value, and adds no value-
[validation](../../../../glossary.md#term-validation) [producer](../../../../glossary.md#term-producer). Execution uses the supplied weights literally and does not clip or
renormalize them. [Tolerance](../../../../glossary.md#term-tolerance) for judging a finite representation's normalized sum, and whether an
execution boundary diagnoses an obligation violation, remain later [conformance](../../../../glossary.md#term-conformance)/execution policy;
this task adds neither. No categorical interpretation or portable special-value guarantee is
promised for a target slice that violates the obligation.

For logits slice `[1, 2, 3]` and dense target `[0, 0, 1]`,
`lse ~= 3.407605964` and the [loss](../../../../glossary.md#loss) is approximately `0.407605964`. For target
`[0.2, 0.3, 0.5]`, the loss is approximately
`0.2 * 2.407605964 + 0.3 * 1.407605964 + 0.5 * 0.407605964 = 1.107605964`.
These are mathematical illustrations; Tensor construction reads none of these values.

Official [PyTorch cross-entropy documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
also describes dense class-probability targets as target-weighted [log-softmax](../../../../glossary.md#softmax--log-softmax) and reduces the
per-[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) output rather than all class [elements](../../../../glossary.md#term-element). Synaptik deliberately selects an arbitrary
explicit class axis, no default, no class weights, no smoothing, no broadcast, and a distinct
dense-target semantic kind. Official
[JAX log-softmax documentation](https://docs.jax.dev/en/latest/_autosummary/jax.nn.log_softmax.html)
documents the log-softmax formula and positive-[infinity](../../../../glossary.md#term-infinity) indeterminacy; the special-value policy
below makes that boundary explicit for this loss.

### Reductions, denominator, and class/[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) empty domains

Let `S` be the logical [element](../../../../glossary.md#term-element) count of the logits [Shape](../../../../glossary.md#shape) after removing the [class axis](../../../../glossary.md#class-axis), once all
unresolved [Dimensions](../../../../glossary.md#dimension) are bound. `S` counts non-class groups, not logits elements, class count,
positive [target](../../../../glossary.md#term-target) count, or target-[weight](../../../../glossary.md#term-weight) sum:

```text
NONE: output[g] = loss[g]                    Shape = logits Shape without class axis
SUM:  output    = sum_g(loss[g])             Shape = scalar
MEAN: output    = sum_g(loss[g]) / S         Shape = scalar
```

[Rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) logits have `S = 1`. A zero [extent](../../../../glossary.md#term-extent) on any non-class axis makes `S = 0`; no class [slice](../../../../glossary.md#slice),
[log-sum-exp](../../../../glossary.md#log-sum-exp), or target weight is evaluated. `NONE` returns an empty result, `SUM` returns [positive
zero](../../../../glossary.md#nan-infinity-and-signed-zero), and [`MEAN`](../../../../glossary.md#term-mean) returns [NaN](../../../../glossary.md#term-nan) from the selected empty mean. Construction stores no `S` attribute
and does not require a statically known count.

A non-empty [sample domain](../../../../glossary.md#sample-domain) requires positive class extent. Class-size behavior is exact:

- `C = 1` is valid. For one [finite](../../../../glossary.md#term-finite) [logit](../../../../glossary.md#logit) and valid target `[1]`, the group [loss](../../../../glossary.md#loss) is positive zero.
- `C = 0` with definitely empty `S = 0` is valid and follows the empty results above.
- `C = 0` with a fully known non-empty sample domain fails locally with:

  ```text
  categoricalCrossEntropyWithLogits class dimension must be positive when sample domain is non-empty: axis=<axis>, dimension=<dimension>
  ```

- when unresolved Dimensions prevent construction from proving either an empty sample domain or
  a positive class extent, construction retains the occurrence with the later obligation
  `S == 0 || C > 0`; [compiler](../../../../glossary.md#term-compiler) [binding](../../../../glossary.md#term-binding) must reject a concrete `S > 0 && C == 0` [occurrence](../../../../glossary.md#term-occurrence).

The local check does not multiply extents: any statically zero non-class Dimension proves
`S = 0`; otherwise all-static non-class Dimensions prove `S > 0`; an unresolved non-class
Dimension leaves the obligation deferred. This avoids construction-time element-count [overflow](../../../../glossary.md#term-overflow).

### Data types, result type, accumulation, and computation format

Logits and [target](../../../../glossary.md#term-target) must each be [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64). Derive the result type exactly once:

```java
DataTypePromotion.promoteFloating(logitsType, targetType)
```

Both operands participate as that promoted type. Every reduction mode returns that result type;
[`MEAN`](../../../../glossary.md#term-mean) does not widen the public result. No cast [producer](../../../../glossary.md#term-producer), hidden conversion [Tensor](../../../../glossary.md#tensor), probability
Tensor, or [log](../../../../glossary.md#term-log)-probability Tensor is inserted.

BFLOAT16 and FLOAT32 results perform [log-sum-exp](../../../../glossary.md#log-sum-exp), subtraction, weighting, class summation, [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)
summation, and mean division in FLOAT32. FLOAT64 results perform them in FLOAT64. Final values are
rounded to the result format. Equal-or-wider intermediates, compensation, vectorization,
parallelization, [fusion](../../../../glossary.md#term-fusion), and [reassociation](../../../../glossary.md#term-reassociation) are permitted when later [conformance](../../../../glossary.md#term-conformance) tolerances and the
exact special-value classes below are preserved. Narrower computation, a fixed traversal/tree,
bitwise [identity](../../../../glossary.md#term-identity-meanings), [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign preservation, and identical [finite](../../../../glossary.md#term-finite) [rounding](../../../../glossary.md#term-rounding) across [backends](../../../../glossary.md#backend)
are not promised.

Exact type failures, checked in input order, are:

```text
categoricalCrossEntropyWithLogits logits must have a floating data type, but was <dataType>
categoricalCrossEntropyWithLogits target must have a floating data type, but was <dataType>
```

### [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed zero](../../../../glossary.md#term-signed-zero), [overflow](../../../../glossary.md#term-overflow), [reassociation](../../../../glossary.md#term-reassociation), and determinism

Construction reads no values. For obligation-satisfying [target](../../../../glossary.md#term-target) slices:

- any NaN [logit](../../../../glossary.md#logit) makes the group [loss](../../../../glossary.md#loss) NaN;
- any positive-infinity logit makes log-softmax indeterminate and the group loss NaN, including a
  single-class slice; no equal-positive-infinity probability split is invented;<span data-glossary-navigation="true"> ([log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [slice glossary](../../../../glossary.md#slice))</span>
- when no logit is positive infinity or NaN, but every logit is negative infinity, the group loss
  is NaN;<span data-glossary-navigation="true"> ([negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- with at least one finite logit and any remaining negative-infinity logits, a positive target
  weight on a negative-infinity logit makes the group loss positive infinity, while a zero weight
  contributes exact positive zero;<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight))</span>
- finite logits use stable log-sum-exp without exponential overflow; finite differences,
  weighting, or accumulation may still overflow to positive infinity or underflow to positive
  zero in the selected computation format;<span data-glossary-navigation="true"> ([log-sum-exp glossary](../../../../glossary.md#log-sum-exp); [finite differences glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks); [underflow glossary](../../../../glossary.md#term-underflow))</span>
- signed-zero logits are ordinary equal finite logits. A valid finite single-class slice returns
  positive zero; every exact zero contribution and an exact-zero finite group loss is positive
  zero; and<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- target NaN, infinity, negativity, or non-normalization violates the caller obligation and has
  no portable categorical-result guarantee.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>

`SUM`/[`MEAN`](../../../../glossary.md#term-mean) propagates any group NaN; otherwise any positive-infinity group makes the reduced
result [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero). An all-[finite](../../../../glossary.md#term-finite) exact-zero non-empty domain reduces to [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero).
Empty `SUM` is positive zero and empty `MEAN` is NaN as specified above. Permitted reassociation
may change finite [rounding](../../../../glossary.md#term-rounding) within later [tolerance](../../../../glossary.md#term-tolerance) but must not change these exact classes.
Separately constructed equal requests remain distinct [producers](../../../../glossary.md#term-producer); there is no interning, [identity](../../../../glossary.md#term-identity-meanings)
determinism, fixed reduction order, or bitwise-result guarantee.

### Result [metadata](../../../../glossary.md#term-metadata), [layout](../../../../glossary.md#layout), [gradients](../../../../glossary.md#term-gradient), [provenance](../../../../glossary.md#provenance), and identifiers

The result [descriptor](../../../../glossary.md#term-descriptor) uses the promoted result type, selected [Shape](../../../../glossary.md#shape), unresolved layout, and
`requiresGrad = logits.requiresGrad() || target.requiresGrad()`. This eligibility is metadata
only and defines no gradient rule. The result is fresh, unlabeled, and storage-free. Neither input
nor its descriptor, label, storage, gradient flag, ID, Shape, provenance, or layout is mutated.

Every success creates one attrs value, one [`Operation`](../../../../glossary.md#operation), one [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), one output
descriptor, one provenance value at output index zero, one [Tensor](../../../../glossary.md#tensor) wrapper, and one Tensor ID.
Provenance records exact ordered inputs `[logits, target]`. No public [softmax](../../../../glossary.md#softmax--log-softmax), [log-softmax](../../../../glossary.md#softmax--log-softmax),
[log-sum-exp](../../../../glossary.md#log-sum-exp), arithmetic, or reduction [producer](../../../../glossary.md#term-producer) is created beneath the [loss](../../../../glossary.md#loss) [occurrence](../../../../glossary.md#term-occurrence).

Every known local [validation](../../../../glossary.md#term-validation) failure occurs before factory delegation and consumes no Tensor ID,
producer, provenance, or wrapper. Factory identifier exhaustion retains the existing
single-output factory behavior; no [rollback](../../../../glossary.md#term-rollback) or new identifier policy is introduced.

### [Validation](../../../../glossary.md#term-validation) and construction order

`TensorLossExpressions.categoricalCrossEntropyWithLogits(...)` performs exactly:

1. null-check logits, [target](../../../../glossary.md#term-target), then reduction;
2. validate logits floating type, then target floating type;
3. read both exact [Shapes](../../../../glossary.md#shape);
4. normalize `classAxis` exactly once against logits Shape;
5. require target [rank](../../../../glossary.md#term-rank) equal logits rank;
6. validate static/deferred positional [Dimension](../../../../glossary.md#dimension) compatibility in increasing logical-[axis](../../../../glossary.md#term-axis) order;
7. apply the local/deferred class-[extent](../../../../glossary.md#term-extent) versus [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)-domain rule;
8. promote logits and target types in [occurrence](../../../../glossary.md#term-occurrence) order;
9. construct attrs with [normalized axis](../../../../glossary.md#normalized-axis) and exact reduction;
10. derive the class-axis-removed logits Shape for `NONE`, or select `Shape.scalar()` for
    `SUM`/[`MEAN`](../../../../glossary.md#term-mean);
11. construct the unresolved result [descriptor](../../../../glossary.md#term-descriptor) and [`Operation`](../../../../glossary.md#operation); and
12. delegate exactly once to the existing single-output derived Tensor factory path with ordered
    inputs <span data-glossary-navigation="true"> ([Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>`[logits, target]`.

Null messages are `logits`, `target`, and `reduction`. The public receiver cannot be null during a
normal Java instance call, but the package-private helper retains explicit logits validation and
tests lock it. The attrs constructor defensively repeats axis and reduction validation. Existing
Shape [normalization](../../../../glossary.md#term-normalization), promotion, descriptor, operation, [producer](../../../../glossary.md#term-producer), and identifier failures retain
their current types and messages.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Mean-squared-error loss](0022-mean-squared-error-loss.md)<span data-glossary-navigation="true"> ([Mean glossary](../../../../glossary.md#term-mean); [loss glossary](../../../../glossary.md#loss))</span>
- [Operation signatures](0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [Typed scalar contract](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- [Softmax semantics](0016i-softmax-semantic-kinds-and-attributes.md)<span data-glossary-navigation="true"> ([Softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- [Softmax Tensor expressions](0016j-softmax-tensor-expressions.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns only backend-independent loss semantics, Tensor metadata, and immutable
  pre-capture provenance. Tensor remains public mutable API state and is not graph IR.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [loss glossary](../../../../glossary.md#loss); [Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [immutable glossary](../../../../glossary.md#term-immutable); [capture glossary](../../../../glossary.md#term-capture-meanings); [provenance glossary](../../../../glossary.md#provenance); [mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Operation types may consume foundational data-type and loss-reduction values but must not import
  Tensor, compiler, training, runtime, prepare, engine, graph state, or backend types.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine))</span>
- Compiler owns graph capture, proof of deferred Shape/class-extent obligations, gradients,
  autograd/backward construction, legal decomposition, canonicalization, and optimization.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [Shape glossary](../../../../glossary.md#shape); [extent glossary](../../../../glossary.md#term-extent); [gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [decomposition glossary](../../../../glossary.md#term-decomposition); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Backend prepare owns stable algorithms, lowering, specialization, fusion, kernels, and
  tolerance satisfaction. Runtime executes prepared work and must not consume <span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion); [kernels glossary](../../../../glossary.md#kernel); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>[`Operation`](../../../../glossary.md#operation) in its
  hot path.<span data-glossary-navigation="true"> ([hot path glossary](../../../../glossary.md#term-hot-path))</span>
- The training extension may later consume the loss Tensor but does not own this semantic
  operation. The task adds no optimizer, parameter, session, step, state, or concrete-backend
  dependency.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [parameter glossary](../../../../glossary.md#term-parameter); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- No architecture, explanatory architecture, ADR, architecture-test, dependency, Gradle,
  cross-module, conformance, integration, compiler, backend, runtime, or training change is
  authorized. Stop if the contract cannot be represented through the current fixed-signature,
  descriptor, and single-output producer foundations.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance); [signature glossary](../../../../glossary.md#term-signature); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer))</span>

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.loss` — owns the new family-specific semantic kind
  variant and immutable class-axis/reduction attributes.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [axis glossary](../../../../glossary.md#term-axis))</span>
- `io.github.pho001.synaptik.model.tensor` — owns the one public fluent receiver plus local
  validation, result Shape, descriptor, and provenance construction.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- existing datatype, [shape](../../../../glossary.md#shape), and [operation](../../../../glossary.md#operation) foundations are consumed without modification.

No package is added, moved, or renamed.

Type placement:

- `...operation.loss.LossKind` — family-owned semantic [identity](../../../../glossary.md#term-identity-meanings) and exact attrs-class [signature](../../../../glossary.md#term-signature).
- `...operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs` — immutable normalized class
  axis and loss reduction because both are intrinsic parameters of this loss meaning.<span data-glossary-navigation="true"> ([class axis glossary](../../../../glossary.md#class-axis); [loss reduction glossary](../../../../glossary.md#loss-reduction); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- `...tensor.TensorLossExpressions` — existing loss-family local validation and construction
  boundary; it remains package-private, final, stateless, and narrowly reusable by task 0022B.
- `...tensor.Tensor` — established public fluent logits receiver [facade](../../../../glossary.md#term-facade).

Tests mirror production packages. The [loss](../../../../glossary.md#loss) semantic test owns enum/signature/attrs contracts; the
same-package [Tensor](../../../../glossary.md#tensor) expression test owns the helper, receiver, Shape, [metadata](../../../../glossary.md#term-metadata), [provenance](../../../../glossary.md#provenance), and
failure contracts.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/LossKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/DenseCategoricalCrossEntropyWithLogitsAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorLossExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (11):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/loss/LossSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`

The nine existing inventory tests change only their exact public [Tensor](../../../../glossary.md#tensor) method count from 186 to
187. No inventory test may receive unrelated assertions or behavior changes.

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime) and Training APIs; [MSE](../../../../glossary.md#mean-squared-error--mse), [softmax](../../../../glossary.md#softmax--log-softmax)/[log-softmax](../../../../glossary.md#softmax--log-softmax),
[log-sum-exp](../../../../glossary.md#log-sum-exp), ordinary reduction, [Shape](../../../../glossary.md#shape), promotion, [signature](../../../../glossary.md#term-signature), factory, [producer](../../../../glossary.md#term-producer), and [provenance](../../../../glossary.md#provenance)
contracts; architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); dependencies; and other
[modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 22 actual paths: four production, eleven tests, and seven documentation/[planning](../../../../glossary.md#term-planning) paths. This
exceeds the planning guide's normal 18-path guardrail under the user's standing higher-path
authorization because the cohesive capability necessarily adds one attrs type, extends one kind
and the existing helper, adds one [Tensor](../../../../glossary.md#tensor) method, adds two focused test responsibilities,
synchronizes the established exact public-count locks atomically, and completes the mandatory
seven-path documentation [handoff](../../../../glossary.md#term-handoff). Splitting an inventory or documentation update would leave the
public [API](../../../../glossary.md#term-application-programming-interface-api) locks broken or the capability incomplete.

The affected-file list is authoritative. If repository evidence shows a different existing
inventory test must change, update this [Ready](../../../../glossary.md#term-ready) task and preserve the exact 22-path maximum before
implementation. Stop for path 23, another public method/type, another test/document, a
factory/foundation modification, any task-0022B implementation/specification, cross-[module](../../../../glossary.md#term-module) work,
architecture or [Gradle](../../../../glossary.md#term-gradle) change, or unrelated cleanup.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document the kind, attrs, helper, and Tensor method: ordered logits/target roles, exact
  Shape/no-broadcast rule, normalized class axis, stable formula, raw target weighting and value
  obligation, output Shape, denominator, class/sample empty policy, types/computation, special
  values, metadata, IDs, failures, and lifecycle boundaries.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [class axis glossary](../../../../glossary.md#class-axis); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Every constructor/method input, non-void result, and expected failure has meaningful `@param`,
  `@return`, and `@throws` text under the [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc profile.
- [Tensor](../../../../glossary.md#tensor) API adds the exact [signature](../../../../glossary.md#term-signature), [axis](../../../../glossary.md#term-axis)/[slice](../../../../glossary.md#slice) [mental model](../../../../glossary.md#term-mental-model), result/reduction table,
  `[1,2,3]` examples, exact/deferred Shape rules, target obligation/no construction-time value
  validation, computation/special values, provenance/IDs, and current-model versus planned
  compiler/backend/runtime/training boundaries.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Compile API records dense logits cross-entropy as current model metadata only. Capture, deferred
  equality and positivity proof, target-obligation policy, gradients, decomposition, validation,
  lowering, and execution remain planned.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering))</span>
- Glossary review covers categorical cross entropy, logits, dense target, class axis, sample
  domain, loss reduction, and mean denominator; update only reusable distinctions.<span data-glossary-navigation="true"> ([categorical cross entropy glossary](../../../../glossary.md#categorical-cross-entropy); [dense target glossary](../../../../glossary.md#dense-target); [sample domain glossary](../../../../glossary.md#sample-domain); [loss reduction glossary](../../../../glossary.md#loss-reduction); [mean glossary](../../../../glossary.md#term-mean))</span>
- Capabilities, master plan, task, and roadmap synchronize 0022A as Complete only after all
  implementation and documentation evidence passes; 0022 remains Complete and 0022B–0024 remain
  Draft without detailed specifications.
- Record reasoned no-change conclusions for Runtime and Training APIs, related contracts,
  architecture/ADRs/tests, conformance/integration, Gradle, dependencies, and other modules.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- `LossKind` has exactly the two ordered values and each resolves only its exact fixed 2/1 attrs
  signature; MSE remains unchanged.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [MSE glossary](../../../../glossary.md#mean-squared-error--mse))</span>
- The new attrs record has exactly normalized [`axis`](../../../../glossary.md#term-axis) then non-null `reduction`, exact validation,
  accessors, equality/hash semantics, and no extra state or surface.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [hash glossary](../../../../glossary.md#term-hash))</span>
- Tensor exposes exactly the one new public method in this task; public Tensor method count is
  exactly 187 and every established inventory lock is synchronized within the exact path list.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Ordered inputs are exactly `[logits, target]`; result provenance uses output index zero and one
  fresh producer/ID on success without public-primitive decomposition.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [producer glossary](../../../../glossary.md#term-producer); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- BFLOAT16/FLOAT32/FLOAT64 pairs promote through current floating promotion; all other inputs fail
  in exact order and no cast is inserted.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- Class axis normalization, exact/static-or-deferred target Shape compatibility, no broadcasting,
  class-axis removal, scalar reduction Shapes, and exact retained Dimension references match this
  task for static, scalar-result, dynamic, zero, and singleton cases.<span data-glossary-navigation="true"> ([Class axis glossary](../../../../glossary.md#class-axis); [normalization glossary](../../../../glossary.md#term-normalization); [target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [broadcasting glossary](../../../../glossary.md#broadcasting); [scalar glossary](../../../../glossary.md#term-scalar); [Dimension glossary](../../../../glossary.md#dimension); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Stable formula, raw target weighting, zero-weight convention, caller value obligation,
  computation formats, denominator, class-size zero/one, empty sample domain, NaN/infinity/
  signed-zero/overflow, reassociation, and determinism match this task without construction-time
  value reads.<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [sample domain glossary](../../../../glossary.md#sample-domain); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Null/type/axis/rank/Dimension/class-extent failures have exact types/messages and occur in the
  specified order before factory delegation with no ID consumption.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [extent glossary](../../../../glossary.md#term-extent))</span>
- Layout is unresolved; gradient eligibility is the input logical OR; results have no label or
  storage; inputs remain unchanged.<span data-glossary-navigation="true"> ([Layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- No index target, ignore index, weight, smoothing, target normalization, gradient, compiler,
  backend, runtime, training, architecture, dependency, or Gradle behavior is added.<span data-glossary-navigation="true"> ([index target glossary](../../../../glossary.md#index-target); [ignore index glossary](../../../../glossary.md#ignore-index); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Exact 22-path maximum and package/type placement hold, or implementation stops to correct the
  Ready plan before proceeding.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- A separate clean documentation-focused agent pass finalizes affected Javadocs, Tensor/Compile
  APIs, glossary impact, capabilities/task/master/roadmap records, official-reference checks, and
  documentation validation in the same overall change.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- Task 0022A becomes `Complete` only after focused/final Java validation, clean docs handoff,
  evidence, empty sections below are completed, and status is synchronized. Task 0022 remains
  Complete; 0022B–0024 remain Draft without detailed specs.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused validation while developing:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.loss.LossSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormInferenceExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLayerNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLinearExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMeanSquaredErrorExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorRmsNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest
```

Final Java validation after executable code stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Documentation-focused pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass also runs targeted Markdown local-link/anchor/fence/final-newline/trailing-
whitespace checks; verifies the official PyTorch and JAX URLs plus both numerical examples;
audits exact 22-path [scope](../../../../glossary.md#term-scope), package placement, exact public [signature](../../../../glossary.md#term-signature)/count, no detailed
0022B–0024 specs, and status [synchronization](../../../../glossary.md#term-synchronization); and records reused Java evidence. It does not rerun
successful Java tests unless it changes executable Java behavior or records a concrete risk.

Repository-wide validation is deferred to the [model](../../../../glossary.md#term-model) [loss](../../../../glossary.md#loss)-family capability [checkpoint](../../../../glossary.md#term-checkpoint) after task
0022B and [CI](../../../../glossary.md#term-continuous-integration-ci) because this is a single-[module](../../../../glossary.md#term-module) [metadata](../../../../glossary.md#term-metadata)-construction task with no [dependency](../../../../glossary.md#term-dependency),
architecture, shared-build, [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), or [runtime](../../../../glossary.md#term-runtime) change.

## Dependencies

- 0001–0002: floating types, promotion, Shape/Dimension equality, static/symbolic extents, axis
  normalization, rank, and scalar Shape.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [rank glossary](../../../../glossary.md#term-rank))</span>
- 0005–0007 and 0011–0013: operation, signature, descriptor, Tensor, factory, provenance, and
  identity foundations.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- 0016I–0016J: first-class softmax/log-softmax axis and slice meaning without requiring public
  decomposition.<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [slice glossary](../../../../glossary.md#slice); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- 0018K: exact attrs-class [signature](../../../../glossary.md#term-signature) and [occurrence](../../../../glossary.md#term-occurrence)-[cardinality](../../../../glossary.md#term-cardinality) [validation](../../../../glossary.md#term-validation).
- 0018L: unified [producer](../../../../glossary.md#term-producer) and output-index [provenance](../../../../glossary.md#provenance).
- 0018N: exact typed-value and receiver-aware validation precedent; no [scalar](../../../../glossary.md#term-scalar) [parameter](../../../../glossary.md#term-parameter) is needed.
- 0018V: log-sum-exp, floating computation, reduction, empty-domain, special-value, and
  determinism precedent.<span data-glossary-navigation="true"> ([log-sum-exp glossary](../../../../glossary.md#log-sum-exp))</span>
- 0022: implemented `LossKind`, `LossReduction`, attrs/helper package ownership, exact-shape
  loss precedent, reduction results, and documentation boundary.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [loss glossary](../../../../glossary.md#loss))</span>

All dependencies are Complete.

## Follow-up tasks

- Draft task 0022B later owns INT32/INT64 index targets, target Shape with the class axis removed,
  optional ignore index, value-bound obligations, and the non-ignored <span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [targets glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [class axis glossary](../../../../glossary.md#class-axis); [ignore index glossary](../../../../glossary.md#ignore-index))</span>[`MEAN`](../../../../glossary.md#term-mean) denominator. It has
  no detailed specification yet and must not reuse the dense attrs type.
- Task 0023 later owns selected compiler-generated backward semantic operations, not autograd
  traversal itself. This task defines no categorical-loss gradient.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [loss glossary](../../../../glossary.md#loss); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Compiler, prepare/runtime, concrete backends, and the training extension later own their
  established capture/proof, validation, autograd, lowering, execution, and coordination work.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backends glossary](../../../../glossary.md#backend); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [lowering glossary](../../../../glossary.md#lowering))</span>
- Task 0024 remains the final [model](../../../../glossary.md#term-model) capability-selection audit.

## Architecture impact

Expected impact: None.

The task adds one [model](../../../../glossary.md#term-model)-owned pure semantic [operation](../../../../glossary.md#operation) variant and one focused attrs value inside
the existing [loss](../../../../glossary.md#loss) package. If implementation requires a new [dependency](../../../../glossary.md#term-dependency), [graph](../../../../glossary.md#graph)-local [identity](../../../../glossary.md#term-identity-meanings),
hidden state, [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend)/training type, [mutable](../../../../glossary.md#term-mutable) service, architecture update, or
different operation-family [ownership](../../../../glossary.md#term-ownership-meanings), stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0022a-dense-target-categorical-cross-entropy-with-logits.md.
Implement that task exactly as specified. Do not implement index targets, ignore index, weights,
label smoothing, gradients/autograd, compiler/backend/runtime/training behavior, or later tasks.
Stop and report any architecture, dependency, affected-file, or exact maximum-scope conflict. Do
not commit or push unless separately authorized.

After Java implementation and recorded focused/final model-test evidence, hand the final diff and
evidence to a separate documentation-focused agent/thread with clean context. That pass must
follow docs/developer-guide/documentation-rules.md, inspect source/tests, finalize Javadocs,
Tensor and Compile APIs, glossary and planning records, run Javadoc/Markdown/official-reference/
scope/status/whitespace checks, and reuse successful Java evidence unless executable Java changes.

Update this task's decisions, limitations, evidence, notes, completion summary, and status. Do not
mark Complete before the documentation pass and every acceptance criterion finishes.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Provide the clean documentation context with this task, the complete diff, focused/final Java
evidence, exact `[logits, target]` roles, attrs/[signature](../../../../glossary.md#term-signature) and public receiver, [Shape](../../../../glossary.md#shape)/[axis](../../../../glossary.md#term-axis),
stable-formula/[target](../../../../glossary.md#term-target)-obligation, reduction/denominator/class/[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)-empty and special-value
contracts, type/[layout](../../../../glossary.md#layout)/[gradient](../../../../glossary.md#term-gradient)/[provenance](../../../../glossary.md#provenance)/ID effects, architecture and exact 22-path limits,
expected [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/glossary/[planning](../../../../glossary.md#term-planning) impact, official PyTorch/JAX links, and required
[Javadoc](../../../../glossary.md#term-javadoc)/Markdown/[scope](../../../../glossary.md#term-scope)/status evidence. It must inspect source/tests independently and record
reasoned no-change conclusions rather than relying on the handoff summary.

## Local decisions

- The public surface is one logits receiver with mandatory class axis and reduction; no convenience
  overload or options object was added.<span data-glossary-navigation="true"> ([class axis glossary](../../../../glossary.md#class-axis))</span>
- Dense targets remain raw exact-shape floating weights with a documented value obligation. Model
  construction deliberately performs metadata validation only.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [shape glossary](../../../../glossary.md#shape); [weights glossary](../../../../glossary.md#term-weight); [Model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation))</span>
- The loss remains one producer rather than a public-primitive decomposition, preserving stable
  target-weighted log-softmax as inspectable semantic meaning.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [producer glossary](../../../../glossary.md#term-producer); [decomposition glossary](../../../../glossary.md#term-decomposition); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- The mean denominator is the non-class sample count. Zero class extent is accepted only when the
  sample domain is definitely empty; unresolved cases retain a later compiler obligation.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [extent glossary](../../../../glossary.md#term-extent); [sample domain glossary](../../../../glossary.md#sample-domain); [compiler glossary](../../../../glossary.md#term-compiler))</span>

## Known limitations

- No class-index target, ignore index, class/sample weight, label smoothing, target normalization,
  gradient rule, compiler capture, backend lowering, runtime execution, or training coordination
  is implemented.<span data-glossary-navigation="true"> ([index target glossary](../../../../glossary.md#index-target); [ignore index glossary](../../../../glossary.md#ignore-index); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [weight glossary](../../../../glossary.md#term-weight); [target glossary](../../../../glossary.md#term-target); [normalization glossary](../../../../glossary.md#term-normalization); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Dynamic [Shape](../../../../glossary.md#shape) equality and `S == 0 || C > 0` are recorded obligations rather than solved during
  Tensor construction.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Numerical examples specify mathematical meaning; this model-only task does not evaluate values
  or establish backend tolerances.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context: the focused command passed 100 tests across 11 requested suites, with
  zero failures, errors, or skips. The final `./gradlew :modules:model:test` passed 954 tests across
  123 suites, with zero failures, errors, or skips. Executable Java did not change afterward; the
  documentation-focused context reused this evidence as required by the documentation workflow.
- Documentation-focused clean context independently reviewed the final four production files,
  eleven test files, generated Javadoc, Tensor/Compile/Training APIs, glossary, task and planning
  records, and related loss, softmax/log-softmax, log-sum-exp, Shape, floating-promotion,
  signature, factory, producer, and provenance contracts.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [planning glossary](../../../../glossary.md#term-planning); [loss glossary](../../../../glossary.md#loss); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-sum-exp glossary](../../../../glossary.md#log-sum-exp); [Shape glossary](../../../../glossary.md#shape); [signature glossary](../../../../glossary.md#term-signature); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- `./gradlew :modules:model:javadoc` passed after final [Javadoc](../../../../glossary.md#term-javadoc) edits.
- Targeted Markdown local-link and anchor, fence, final-newline, and trailing-whitespace checks
  passed for all changed documentation. Official PyTorch cross-entropy and JAX log-softmax URLs
  resolved, and the `[1, 2, 3]` one-hot and dense-[target](../../../../glossary.md#term-target) examples were independently recalculated.
- Generated-Javadoc/public-surface/package audits confirmed the new attrs and Tensor method, exact
  ordered `[logits, target]` inputs, exact Tensor public-method count 187, and no unintended public
  method or package change. Scope audit found exactly 22 paths: four production, eleven tests, and
  seven documentation/planning paths. No detailed 0022B–0024 task specification exists.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>
- Status [synchronization](../../../../glossary.md#term-synchronization), exact-path, formatting, and `git diff --check` audits passed.
- A final implementation-agent audit identified that the first Javadoc pass named stable
  log-softmax and special-value classes without spelling out their full contract. The same clean
  documentation context corrected that gap across the kind, attrs, construction helper, and
  public Tensor method without changing executable behavior. Fresh model Javadoc generation and
  source/rendered-content, exact-scope/status, and whitespace audits passed afterward.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Runtime and Training API pages require no change because they expose no implemented loss
  execution or session behavior. Architecture/ADRs/tests, conformance/integration, Gradle,
  dependencies, and other modules require no change because this task adds model-owned metadata
  inside existing boundaries without dependency, build, execution, or cross-module behavior.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [metadata glossary](../../../../glossary.md#term-metadata); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Implementation notes

- Added one loss kind and exact attrs signature, one immutable attrs record, one package-private
  construction path, and one public Tensor receiver.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [signature glossary](../../../../glossary.md#term-signature); [immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Local construction validates nulls, floating types, axis, exact/deferred Shape compatibility,
  and statically provable class/sample emptiness before one factory delegation. Successful results
  preserve exact input provenance, promoted type, selected Shape, unresolved layout, and combined
  gradient-eligibility metadata.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [provenance glossary](../../../../glossary.md#provenance); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Existing public-method inventory tests were synchronized within their owning eleven test paths;
  no unrelated production behavior changed.

## Completion summary

- Completed dense-target categorical cross-entropy-with-logits model semantics and the one public
  Tensor expression with stable target-weighted log-softmax meaning, exact Shape rules, explicit
  reduction, sample-count denominator, special-value policy, and metadata/provenance contracts.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [Shape glossary](../../../../glossary.md#shape); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Finalized all affected production Javadocs plus Tensor API, Compile API, glossary, capabilities,
  task, master plan, and roadmap documentation.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile))</span>
- Validation passed with reused 100-test focused and 954-test final model evidence, fresh model
  Javadoc generation, Markdown/reference/example/public-surface/package/scope/status audits, and<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check`.
- Unresolved issues: none. Required follow-up: none for task 0022A; task 0022B remains separate
  Draft future work without a detailed specification.

Status: Complete
