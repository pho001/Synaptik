# Task 0022A: Dense-Target Categorical Cross-Entropy with Logits

## Status

Complete

## Goal

Add one intentionally narrow backend-independent dense-target categorical cross-entropy meaning
that consumes logits and an exact-shape floating target, normalizes one explicit class axis, and
applies the existing explicit `NONE`, `SUM`, or `MEAN` loss reduction.

Mental model:

```text
logits + exact-shape dense probability target + class axis + explicit reduction
  -> one loss producer
     -> one weighted stable-log-softmax loss per non-class coordinate
        -> non-class Shape or one reduced scalar
```

The operation remains one inspectable request. Model construction must not decompose it into
public log-softmax, multiplication, or reduction expressions.

## Scope

- Append exactly `DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS` to `LossKind` after
  `MEAN_SQUARED_ERROR`.
- Add immutable public
  `DenseCategoricalCrossEntropyWithLogitsAttrs(int axis, LossReduction reduction)`.
- Give the new kind exactly one fixed two-input/one-output signature while preserving the
  mean-squared-error kind and signature unchanged.
- Add exactly this fluent logits-receiver method:

  ```java
  public Tensor categoricalCrossEntropyWithLogits(
          Tensor target, int classAxis, LossReduction reduction)
  ```

- Extend the existing package-private, final, field-free `TensorLossExpressions` helper with one
  dense-target entry and only the narrowly shared private validation/Shape helpers justified by
  both loss constructions.
- Fix ordered roles, exact Shape compatibility, normalized class-axis metadata, result Shape,
  floating promotion, stable logits formula, dense-target value obligation, denominator, empty
  domains, special values, provenance, identifiers, diagnostics, and validation order.
- Add focused semantic and Tensor-expression coverage and synchronize every existing exact public
  Tensor method-count inventory from 186 to 187.
- Finalize affected Javadocs, Tensor and Compile API references, glossary impact, capabilities,
  task/master/roadmap records, and documentation checks through the mandatory clean
  documentation-focused handoff.

## Out of scope

- index targets, integer labels, class-axis-removed target input, ignore index, bounds checking,
  optional log-probability output, or any task-0022B implementation or detailed specification
- class weights, sample weights, masks, label smoothing, temperature, scale, epsilon, probability
  clipping, target renormalization, target broadcasting, implicit expansion, implicit casts,
  probability-input cross entropy, standalone negative-log-likelihood, binary cross entropy,
  Kullback–Leibler divergence, or another loss
- a default axis, default reduction, trailing-axis convenience, overload, static facade, alias,
  options object, parser, registry, primitive/string reduction spelling, or eager constant
- value/storage reads during model construction, eager target validation, tolerances, a fixed
  max/sum traversal or reduction tree, public-primitive decomposition, or a backend algorithm
- gradients, adjoints, autograd traversal, compiler capture/validation/decomposition/fusion,
  backend support/lowering/kernels, prepare, runtime, publication, execution, or training-session,
  optimizer, parameter, module, step, and checkpoint behavior
- FLOAT16, integral, BOOL, complex, sparse, quantized, unsigned, or nullable operands/results
- changes to `LossReduction`, softmax/log-softmax, log-sum-exp, aggregate reductions, Shape,
  floating promotion, producer/provenance, factory, compiler/runtime/training Java APIs or
  behavior, architecture, dependencies, Gradle, another module, task 0022, or tasks 0023–0024

## Public and operation contracts

### Kind, typed attributes, and signature

`LossKind` contains exactly these values in order:

```java
MEAN_SQUARED_ERROR,
DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
```

The new kind's exact ordered signature is:

```java
OperationSignature.fixed(DenseCategoricalCrossEntropyWithLogitsAttrs.class, 2, 1)
```

One occurrence has ordered inputs `[logits, target]` and ordered output `[loss]`. The receiver is
the logits input at position zero. There is no hidden log-probability input/output, saved value,
sibling result, or variable cardinality. `MEAN_SQUARED_ERROR` continues to accept only
`MeanSquaredErrorAttrs` with its current fixed 2/1 signature.

`DenseCategoricalCrossEntropyWithLogitsAttrs` has exactly these components and order:

```java
int axis,
LossReduction reduction
```

The stored axis is already normalized to a non-negative logits axis. Its compact constructor:

1. rejects a negative axis with
   `IllegalArgumentException("axis must be non-negative: " + axis)`; then
2. null-checks reduction with `NullPointerException("reduction")`.

It retains the exact enum reference and stores no Tensor, Shape, class extent, sample count,
denominator, target-normalization flag, value, data type, algorithm, gradient, graph, compiler,
backend, runtime, or training state. It reuses `LossReduction` as a configuration component and
does not change that enum's members or meaning.

### Public receiver, ranks, class axis, and target Shape

The public surface is exactly:

```java
public Tensor categoricalCrossEntropyWithLogits(
        Tensor target, int classAxis, LossReduction reduction)
```

The receiver is logits. Logits must have rank at least one because a scalar has no class axis.
`classAxis` accepts the current Shape range `[-rank, rank - 1]`, is normalized exactly once with
`logitsShape.normalizeAxis(classAxis)`, and only the normalized non-negative value is stored.

Target rank and every corresponding Dimension must match logits positionally:

- structurally equal Dimensions pass;
- unequal static Dimensions fail locally;
- a non-structurally-equal pair involving an unresolved Dimension is accepted with an equality
  obligation for later compiler capture/binding; and
- no scalar, singleton, leading-axis, right-aligned, or other broadcasting is accepted.

Result Shape is derived only from logits:

- `NONE` removes the normalized class axis and retains every other logits Dimension by exact
  reference and in order; removing the only axis returns canonical `Shape.scalar()`; and
- `SUM` and `MEAN` return canonical `Shape.scalar()`.

The target Shape object is never substituted into the result. Exact Shape failures are:

```text
categoricalCrossEntropyWithLogits target rank must equal logits rank: logits=<logitsRank>, target=<targetRank>
categoricalCrossEntropyWithLogits target dimension mismatch at axis <axis>: logits=<logitsDimension>, target=<targetDimension>
```

An invalid raw axis retains the current Shape failure type and exact message:

```text
Axis <classAxis> is outside shape rank <logitsRank>
```

### Stable logits formula and dense-target obligation

For each non-class coordinate `g`, let `z[g,c]` be the logits slice, `t[g,c]` be the supplied
dense target weight, and `C` be the eventual class extent. In the selected computation format:

```text
m[g]   = max_c(z[g,c])
lse[g] = m[g] + log(sum_c(exp(z[g,c] - m[g])))
q[g,c] = lse[g] - z[g,c]
loss[g] = sum_c(weightedContribution(t[g,c], q[g,c]))

weightedContribution(0, q) = +0
weightedContribution(t, q) = t * q, for t > 0
```

This is the stable-logits form of `-sum_c(t[g,c] * logSoftmax(z[g,:])[c])`. It must not first
materialize probabilities, take their logarithm, or expand into public Tensor primitives. A later
backend may use any conforming fused or decomposed stable algorithm, but naive `exp(z)` followed
by normalization is not a conforming finite-logit implementation when it overflows and changes
the selected result class. The explicit zero-weight convention gives an absent class exactly
positive-zero contribution even when its log probability is negative infinity.

For every evaluated group, callers have the value obligation that target entries are finite and
non-negative and describe a normalized probability distribution along the class axis. Model
construction deliberately reads no target values, rejects no target value, and adds no value-
validation producer. Execution uses the supplied weights literally and does not clip or
renormalize them. Tolerance for judging a finite representation's normalized sum, and whether an
execution boundary diagnoses an obligation violation, remain later conformance/execution policy;
this task adds neither. No categorical interpretation or portable special-value guarantee is
promised for a target slice that violates the obligation.

For logits slice `[1, 2, 3]` and dense target `[0, 0, 1]`,
`lse ~= 3.407605964` and the loss is approximately `0.407605964`. For target
`[0.2, 0.3, 0.5]`, the loss is approximately
`0.2 * 2.407605964 + 0.3 * 1.407605964 + 0.5 * 0.407605964 = 1.107605964`.
These are mathematical illustrations; Tensor construction reads none of these values.

Official [PyTorch cross-entropy documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
also describes dense class-probability targets as target-weighted log-softmax and reduces the
per-sample output rather than all class elements. Synaptik deliberately selects an arbitrary
explicit class axis, no default, no class weights, no smoothing, no broadcast, and a distinct
dense-target semantic kind. Official
[JAX log-softmax documentation](https://docs.jax.dev/en/latest/_autosummary/jax.nn.log_softmax.html)
documents the log-softmax formula and positive-infinity indeterminacy; the special-value policy
below makes that boundary explicit for this loss.

### Reductions, denominator, and class/sample empty domains

Let `S` be the logical element count of the logits Shape after removing the class axis, once all
unresolved Dimensions are bound. `S` counts non-class groups, not logits elements, class count,
positive target count, or target-weight sum:

```text
NONE: output[g] = loss[g]                    Shape = logits Shape without class axis
SUM:  output    = sum_g(loss[g])             Shape = scalar
MEAN: output    = sum_g(loss[g]) / S         Shape = scalar
```

Rank-one logits have `S = 1`. A zero extent on any non-class axis makes `S = 0`; no class slice,
log-sum-exp, or target weight is evaluated. `NONE` returns an empty result, `SUM` returns positive
zero, and `MEAN` returns NaN from the selected empty mean. Construction stores no `S` attribute
and does not require a statically known count.

A non-empty sample domain requires positive class extent. Class-size behavior is exact:

- `C = 1` is valid. For one finite logit and valid target `[1]`, the group loss is positive zero.
- `C = 0` with definitely empty `S = 0` is valid and follows the empty results above.
- `C = 0` with a fully known non-empty sample domain fails locally with:

  ```text
  categoricalCrossEntropyWithLogits class dimension must be positive when sample domain is non-empty: axis=<axis>, dimension=<dimension>
  ```

- when unresolved Dimensions prevent construction from proving either an empty sample domain or
  a positive class extent, construction retains the occurrence with the later obligation
  `S == 0 || C > 0`; compiler binding must reject a concrete `S > 0 && C == 0` occurrence.

The local check does not multiply extents: any statically zero non-class Dimension proves
`S = 0`; otherwise all-static non-class Dimensions prove `S > 0`; an unresolved non-class
Dimension leaves the obligation deferred. This avoids construction-time element-count overflow.

### Data types, result type, accumulation, and computation format

Logits and target must each be BFLOAT16, FLOAT32, or FLOAT64. Derive the result type exactly once:

```java
DataTypePromotion.promoteFloating(logitsType, targetType)
```

Both operands participate as that promoted type. Every reduction mode returns that result type;
`MEAN` does not widen the public result. No cast producer, hidden conversion Tensor, probability
Tensor, or log-probability Tensor is inserted.

BFLOAT16 and FLOAT32 results perform log-sum-exp, subtraction, weighting, class summation, sample
summation, and mean division in FLOAT32. FLOAT64 results perform them in FLOAT64. Final values are
rounded to the result format. Equal-or-wider intermediates, compensation, vectorization,
parallelization, fusion, and reassociation are permitted when later conformance tolerances and the
exact special-value classes below are preserved. Narrower computation, a fixed traversal/tree,
bitwise identity, NaN payload/sign preservation, and identical finite rounding across backends
are not promised.

Exact type failures, checked in input order, are:

```text
categoricalCrossEntropyWithLogits logits must have a floating data type, but was <dataType>
categoricalCrossEntropyWithLogits target must have a floating data type, but was <dataType>
```

### NaN, infinity, signed zero, overflow, reassociation, and determinism

Construction reads no values. For obligation-satisfying target slices:

- any NaN logit makes the group loss NaN;
- any positive-infinity logit makes log-softmax indeterminate and the group loss NaN, including a
  single-class slice; no equal-positive-infinity probability split is invented;
- when no logit is positive infinity or NaN, but every logit is negative infinity, the group loss
  is NaN;
- with at least one finite logit and any remaining negative-infinity logits, a positive target
  weight on a negative-infinity logit makes the group loss positive infinity, while a zero weight
  contributes exact positive zero;
- finite logits use stable log-sum-exp without exponential overflow; finite differences,
  weighting, or accumulation may still overflow to positive infinity or underflow to positive
  zero in the selected computation format;
- signed-zero logits are ordinary equal finite logits. A valid finite single-class slice returns
  positive zero; every exact zero contribution and an exact-zero finite group loss is positive
  zero; and
- target NaN, infinity, negativity, or non-normalization violates the caller obligation and has
  no portable categorical-result guarantee.

`SUM`/`MEAN` propagates any group NaN; otherwise any positive-infinity group makes the reduced
result positive infinity. An all-finite exact-zero non-empty domain reduces to positive zero.
Empty `SUM` is positive zero and empty `MEAN` is NaN as specified above. Permitted reassociation
may change finite rounding within later tolerance but must not change these exact classes.
Separately constructed equal requests remain distinct producers; there is no interning, identity
determinism, fixed reduction order, or bitwise-result guarantee.

### Result metadata, layout, gradients, provenance, and identifiers

The result descriptor uses the promoted result type, selected Shape, unresolved layout, and
`requiresGrad = logits.requiresGrad() || target.requiresGrad()`. This eligibility is metadata
only and defines no gradient rule. The result is fresh, unlabeled, and storage-free. Neither input
nor its descriptor, label, storage, gradient flag, ID, Shape, provenance, or layout is mutated.

Every success creates one attrs value, one `Operation`, one `TensorProducer`, one output
descriptor, one provenance value at output index zero, one Tensor wrapper, and one Tensor ID.
Provenance records exact ordered inputs `[logits, target]`. No public softmax, log-softmax,
log-sum-exp, arithmetic, or reduction producer is created beneath the loss occurrence.

Every known local validation failure occurs before factory delegation and consumes no Tensor ID,
producer, provenance, or wrapper. Factory identifier exhaustion retains the existing
single-output factory behavior; no rollback or new identifier policy is introduced.

### Validation and construction order

`TensorLossExpressions.categoricalCrossEntropyWithLogits(...)` performs exactly:

1. null-check logits, target, then reduction;
2. validate logits floating type, then target floating type;
3. read both exact Shapes;
4. normalize `classAxis` exactly once against logits Shape;
5. require target rank equal logits rank;
6. validate static/deferred positional Dimension compatibility in increasing logical-axis order;
7. apply the local/deferred class-extent versus sample-domain rule;
8. promote logits and target types in occurrence order;
9. construct attrs with normalized axis and exact reduction;
10. derive the class-axis-removed logits Shape for `NONE`, or select `Shape.scalar()` for
    `SUM`/`MEAN`;
11. construct the unresolved result descriptor and `Operation`; and
12. delegate exactly once to the existing single-output derived Tensor factory path with ordered
    inputs `[logits, target]`.

Null messages are `logits`, `target`, and `reduction`. The public receiver cannot be null during a
normal Java instance call, but the package-private helper retains explicit logits validation and
tests lock it. The attrs constructor defensively repeats axis and reduction validation. Existing
Shape normalization, promotion, descriptor, operation, producer, and identifier failures retain
their current types and messages.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Mean-squared-error loss](0022-mean-squared-error-loss.md)
- [Operation signatures](0018k-operation-signature-and-construction-hardening.md)
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)
- [Typed scalar contract](0018n-typed-scalar-value-contract.md)
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)
- [Softmax semantics](0016i-softmax-semantic-kinds-and-attributes.md)
- [Softmax Tensor expressions](0016j-softmax-tensor-expressions.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns only backend-independent loss semantics, Tensor metadata, and immutable
  pre-capture provenance. Tensor remains public mutable API state and is not graph IR.
- Operation types may consume foundational data-type and loss-reduction values but must not import
  Tensor, compiler, training, runtime, prepare, engine, graph state, or backend types.
- Compiler owns graph capture, proof of deferred Shape/class-extent obligations, gradients,
  autograd/backward construction, legal decomposition, canonicalization, and optimization.
- Backend prepare owns stable algorithms, lowering, specialization, fusion, kernels, and
  tolerance satisfaction. Runtime executes prepared work and must not consume `Operation` in its
  hot path.
- The training extension may later consume the loss Tensor but does not own this semantic
  operation. The task adds no optimizer, parameter, session, step, state, or concrete-backend
  dependency.
- No architecture, explanatory architecture, ADR, architecture-test, dependency, Gradle,
  cross-module, conformance, integration, compiler, backend, runtime, or training change is
  authorized. Stop if the contract cannot be represented through the current fixed-signature,
  descriptor, and single-output producer foundations.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.loss` — owns the new family-specific semantic kind
  variant and immutable class-axis/reduction attributes.
- `io.github.pho001.synaptik.model.tensor` — owns the one public fluent receiver plus local
  validation, result Shape, descriptor, and provenance construction.
- existing datatype, shape, and operation foundations are consumed without modification.

No package is added, moved, or renamed.

Type placement:

- `...operation.loss.LossKind` — family-owned semantic identity and exact attrs-class signature.
- `...operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs` — immutable normalized class
  axis and loss reduction because both are intrinsic parameters of this loss meaning.
- `...tensor.TensorLossExpressions` — existing loss-family local validation and construction
  boundary; it remains package-private, final, stateless, and narrowly reusable by task 0022B.
- `...tensor.Tensor` — established public fluent logits receiver facade.

Tests mirror production packages. The loss semantic test owns enum/signature/attrs contracts; the
same-package Tensor expression test owns the helper, receiver, Shape, metadata, provenance, and
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

The nine existing inventory tests change only their exact public Tensor method count from 186 to
187. No inventory test may receive unrelated assertions or behavior changes.

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Runtime and Training APIs; MSE, softmax/log-softmax,
log-sum-exp, ordinary reduction, Shape, promotion, signature, factory, producer, and provenance
contracts; architecture/ADRs/tests; conformance/integration; Gradle; dependencies; and other
modules.

## Maximum scope

Exactly 22 actual paths: four production, eleven tests, and seven documentation/planning paths. This
exceeds the planning guide's normal 18-path guardrail under the user's standing higher-path
authorization because the cohesive capability necessarily adds one attrs type, extends one kind
and the existing helper, adds one Tensor method, adds two focused test responsibilities,
synchronizes the established exact public-count locks atomically, and completes the mandatory
seven-path documentation handoff. Splitting an inventory or documentation update would leave the
public API locks broken or the capability incomplete.

The affected-file list is authoritative. If repository evidence shows a different existing
inventory test must change, update this Ready task and preserve the exact 22-path maximum before
implementation. Stop for path 23, another public method/type, another test/document, a
factory/foundation modification, any task-0022B implementation/specification, cross-module work,
architecture or Gradle change, or unrelated cleanup.

## Javadoc and documentation requirements

- Fully document the kind, attrs, helper, and Tensor method: ordered logits/target roles, exact
  Shape/no-broadcast rule, normalized class axis, stable formula, raw target weighting and value
  obligation, output Shape, denominator, class/sample empty policy, types/computation, special
  values, metadata, IDs, failures, and lifecycle boundaries.
- Every constructor/method input, non-void result, and expected failure has meaningful `@param`,
  `@return`, and `@throws` text under the API/Javadoc profile.
- Tensor API adds the exact signature, axis/slice mental model, result/reduction table,
  `[1,2,3]` examples, exact/deferred Shape rules, target obligation/no construction-time value
  validation, computation/special values, provenance/IDs, and current-model versus planned
  compiler/backend/runtime/training boundaries.
- Compile API records dense logits cross-entropy as current model metadata only. Capture, deferred
  equality and positivity proof, target-obligation policy, gradients, decomposition, validation,
  lowering, and execution remain planned.
- Glossary review covers categorical cross entropy, logits, dense target, class axis, sample
  domain, loss reduction, and mean denominator; update only reusable distinctions.
- Capabilities, master plan, task, and roadmap synchronize 0022A as Complete only after all
  implementation and documentation evidence passes; 0022 remains Complete and 0022B–0024 remain
  Draft without detailed specifications.
- Record reasoned no-change conclusions for Runtime and Training APIs, related contracts,
  architecture/ADRs/tests, conformance/integration, Gradle, dependencies, and other modules.

## Acceptance criteria

- `LossKind` has exactly the two ordered values and each resolves only its exact fixed 2/1 attrs
  signature; MSE remains unchanged.
- The new attrs record has exactly normalized `axis` then non-null `reduction`, exact validation,
  accessors, equality/hash semantics, and no extra state or surface.
- Tensor exposes exactly the one new public method in this task; public Tensor method count is
  exactly 187 and every established inventory lock is synchronized within the exact path list.
- Ordered inputs are exactly `[logits, target]`; result provenance uses output index zero and one
  fresh producer/ID on success without public-primitive decomposition.
- BFLOAT16/FLOAT32/FLOAT64 pairs promote through current floating promotion; all other inputs fail
  in exact order and no cast is inserted.
- Class axis normalization, exact/static-or-deferred target Shape compatibility, no broadcasting,
  class-axis removal, scalar reduction Shapes, and exact retained Dimension references match this
  task for static, scalar-result, dynamic, zero, and singleton cases.
- Stable formula, raw target weighting, zero-weight convention, caller value obligation,
  computation formats, denominator, class-size zero/one, empty sample domain, NaN/infinity/
  signed-zero/overflow, reassociation, and determinism match this task without construction-time
  value reads.
- Null/type/axis/rank/Dimension/class-extent failures have exact types/messages and occur in the
  specified order before factory delegation with no ID consumption.
- Layout is unresolved; gradient eligibility is the input logical OR; results have no label or
  storage; inputs remain unchanged.
- No index target, ignore index, weight, smoothing, target normalization, gradient, compiler,
  backend, runtime, training, architecture, dependency, or Gradle behavior is added.
- Exact 22-path maximum and package/type placement hold, or implementation stops to correct the
  Ready plan before proceeding.
- A separate clean documentation-focused agent pass finalizes affected Javadocs, Tensor/Compile
  APIs, glossary impact, capabilities/task/master/roadmap records, official-reference checks, and
  documentation validation in the same overall change.
- Task 0022A becomes `Complete` only after focused/final Java validation, clean docs handoff,
  evidence, empty sections below are completed, and status is synchronized. Task 0022 remains
  Complete; 0022B–0024 remain Draft without detailed specs.

## Tests / validation

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
audits exact 22-path scope, package placement, exact public signature/count, no detailed
0022B–0024 specs, and status synchronization; and records reused Java evidence. It does not rerun
successful Java tests unless it changes executable Java behavior or records a concrete risk.

Repository-wide validation is deferred to the model loss-family capability checkpoint after task
0022B and CI because this is a single-module metadata-construction task with no dependency,
architecture, shared-build, compiler, backend, or runtime change.

## Dependencies

- 0001–0002: floating types, promotion, Shape/Dimension equality, static/symbolic extents, axis
  normalization, rank, and scalar Shape.
- 0005–0007 and 0011–0013: operation, signature, descriptor, Tensor, factory, provenance, and
  identity foundations.
- 0016I–0016J: first-class softmax/log-softmax axis and slice meaning without requiring public
  decomposition.
- 0018K: exact attrs-class signature and occurrence-cardinality validation.
- 0018L: unified producer and output-index provenance.
- 0018N: exact typed-value and receiver-aware validation precedent; no scalar parameter is needed.
- 0018V: log-sum-exp, floating computation, reduction, empty-domain, special-value, and
  determinism precedent.
- 0022: implemented `LossKind`, `LossReduction`, attrs/helper package ownership, exact-shape
  loss precedent, reduction results, and documentation boundary.

All dependencies are Complete.

## Follow-up tasks

- Draft task 0022B later owns INT32/INT64 index targets, target Shape with the class axis removed,
  optional ignore index, value-bound obligations, and the non-ignored `MEAN` denominator. It has
  no detailed specification yet and must not reuse the dense attrs type.
- Task 0023 later owns selected compiler-generated backward semantic operations, not autograd
  traversal itself. This task defines no categorical-loss gradient.
- Compiler, prepare/runtime, concrete backends, and the training extension later own their
  established capture/proof, validation, autograd, lowering, execution, and coordination work.
- Task 0024 remains the final model capability-selection audit.

## Architecture impact

Expected impact: None.

The task adds one model-owned pure semantic operation variant and one focused attrs value inside
the existing loss package. If implementation requires a new dependency, graph-local identity,
hidden state, compiler/runtime/backend/training type, mutable service, architecture update, or
different operation-family ownership, stop and report the conflict.

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

## Documentation-agent handoff

Provide the clean documentation context with this task, the complete diff, focused/final Java
evidence, exact `[logits, target]` roles, attrs/signature and public receiver, Shape/axis,
stable-formula/target-obligation, reduction/denominator/class/sample-empty and special-value
contracts, type/layout/gradient/provenance/ID effects, architecture and exact 22-path limits,
expected Tensor/Compile/glossary/planning impact, official PyTorch/JAX links, and required
Javadoc/Markdown/scope/status evidence. It must inspect source/tests independently and record
reasoned no-change conclusions rather than relying on the handoff summary.

## Local decisions

- The public surface is one logits receiver with mandatory class axis and reduction; no convenience
  overload or options object was added.
- Dense targets remain raw exact-shape floating weights with a documented value obligation. Model
  construction deliberately performs metadata validation only.
- The loss remains one producer rather than a public-primitive decomposition, preserving stable
  target-weighted log-softmax as inspectable semantic meaning.
- The mean denominator is the non-class sample count. Zero class extent is accepted only when the
  sample domain is definitely empty; unresolved cases retain a later compiler obligation.

## Known limitations

- No class-index target, ignore index, class/sample weight, label smoothing, target normalization,
  gradient rule, compiler capture, backend lowering, runtime execution, or training coordination
  is implemented.
- Dynamic Shape equality and `S == 0 || C > 0` are recorded obligations rather than solved during
  Tensor construction.
- Numerical examples specify mathematical meaning; this model-only task does not evaluate values
  or establish backend tolerances.

## Validation evidence

- Implementation context: the focused command passed 100 tests across 11 requested suites, with
  zero failures, errors, or skips. The final `./gradlew :modules:model:test` passed 954 tests across
  123 suites, with zero failures, errors, or skips. Executable Java did not change afterward; the
  documentation-focused context reused this evidence as required by the documentation workflow.
- Documentation-focused clean context independently reviewed the final four production files,
  eleven test files, generated Javadoc, Tensor/Compile/Training APIs, glossary, task and planning
  records, and related loss, softmax/log-softmax, log-sum-exp, Shape, floating-promotion,
  signature, factory, producer, and provenance contracts.
- `./gradlew :modules:model:javadoc` passed after final Javadoc edits.
- Targeted Markdown local-link and anchor, fence, final-newline, and trailing-whitespace checks
  passed for all changed documentation. Official PyTorch cross-entropy and JAX log-softmax URLs
  resolved, and the `[1, 2, 3]` one-hot and dense-target examples were independently recalculated.
- Generated-Javadoc/public-surface/package audits confirmed the new attrs and Tensor method, exact
  ordered `[logits, target]` inputs, exact Tensor public-method count 187, and no unintended public
  method or package change. Scope audit found exactly 22 paths: four production, eleven tests, and
  seven documentation/planning paths. No detailed 0022B–0024 task specification exists.
- Status synchronization, exact-path, formatting, and `git diff --check` audits passed.
- A final implementation-agent audit identified that the first Javadoc pass named stable
  log-softmax and special-value classes without spelling out their full contract. The same clean
  documentation context corrected that gap across the kind, attrs, construction helper, and
  public Tensor method without changing executable behavior. Fresh model Javadoc generation and
  source/rendered-content, exact-scope/status, and whitespace audits passed afterward.
- Runtime and Training API pages require no change because they expose no implemented loss
  execution or session behavior. Architecture/ADRs/tests, conformance/integration, Gradle,
  dependencies, and other modules require no change because this task adds model-owned metadata
  inside existing boundaries without dependency, build, execution, or cross-module behavior.

## Implementation notes

- Added one loss kind and exact attrs signature, one immutable attrs record, one package-private
  construction path, and one public Tensor receiver.
- Local construction validates nulls, floating types, axis, exact/deferred Shape compatibility,
  and statically provable class/sample emptiness before one factory delegation. Successful results
  preserve exact input provenance, promoted type, selected Shape, unresolved layout, and combined
  gradient-eligibility metadata.
- Existing public-method inventory tests were synchronized within their owning eleven test paths;
  no unrelated production behavior changed.

## Completion summary

- Completed dense-target categorical cross-entropy-with-logits model semantics and the one public
  Tensor expression with stable target-weighted log-softmax meaning, exact Shape rules, explicit
  reduction, sample-count denominator, special-value policy, and metadata/provenance contracts.
- Finalized all affected production Javadocs plus Tensor API, Compile API, glossary, capabilities,
  task, master plan, and roadmap documentation.
- Validation passed with reused 100-test focused and 954-test final model evidence, fresh model
  Javadoc generation, Markdown/reference/example/public-surface/package/scope/status audits, and
  `git diff --check`.
- Unresolved issues: none. Required follow-up: none for task 0022A; task 0022B remains separate
  Draft future work without a detailed specification.

Status: Complete
