# Task 0022: [Mean](../../../../glossary.md#term-mean)-Squared-Error [Loss](../../../../glossary.md#loss)

## Status

Complete

## Goal

Add the first intentionally selected [loss](../../../../glossary.md#loss) [operation](../../../../glossary.md#operation): one [backend](../../../../glossary.md#backend)-independent [mean](../../../../glossary.md#term-mean)-squared-error
([MSE](../../../../glossary.md#mean-squared-error--mse)) meaning and one public [Tensor](../../../../glossary.md#tensor) expression with explicit `NONE`, `SUM`, or `MEAN` reduction.
The task establishes reusable typed loss-reduction vocabulary without implementing categorical
losses, [gradients](../../../../glossary.md#term-gradient), training coordination, [compiler](../../../../glossary.md#term-compiler) behavior, or execution.

[Mental model](../../../../glossary.md#term-mental-model):

```text
prediction + exact-shape target + explicit reduction
  -> one loss producer
     -> elementwise squared error or one reduced scalar
```

## [Scope](../../../../glossary.md#term-scope)

- Add public `LossKind.MEAN_SQUARED_ERROR` in a focused [model](../../../../glossary.md#term-model)-owned [loss](../../../../glossary.md#loss) package.
- Add public `LossReduction` with exactly `NONE`, `SUM`, and [`MEAN`](../../../../glossary.md#term-mean), in that order.
- Add [immutable](../../../../glossary.md#term-immutable) public `MeanSquaredErrorAttrs(LossReduction reduction)`.
- Give the [operation](../../../../glossary.md#operation) exactly one fixed two-input/one-output [signature](../../../../glossary.md#term-signature).
- Add exactly this fluent receiver method:

  ```java
  public Tensor meanSquaredError(Tensor target, LossReduction reduction)
  ```

- Add one package-private, final, field-free `TensorLossExpressions` construction helper.
- Fix ordered roles, Shape/type validation, result metadata, formula, denominator and empty-domain
  policy, special-value classes, provenance, identifier effects, diagnostics, and validation order.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Finalize affected Javadocs, Tensor and Compile API references, glossary impact, and planning
  records in the mandatory clean documentation-focused handoff.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [handoff glossary](../../../../glossary.md#term-handoff))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- dense-target or index-target categorical cross-entropy, caller-supplied probability or
  log-probability losses, negative-log-likelihood as a separate operation, binary cross-entropy,
  Kullback–Leibler divergence, margin/ranking, connectionist temporal classification, or any other
  loss<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking))</span><span data-glossary-navigation="true"> ([log glossary](../../../../glossary.md#term-log); [losses glossary](../../../../glossary.md#loss); [operation glossary](../../../../glossary.md#operation); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- class axes, class/sample/element weights, masks, ignore index, label smoothing, normalization
  factors other than the selected logical element count, or a loss registry/options framework<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [element glossary](../../../../glossary.md#term-element); [weights glossary](../../../../glossary.md#term-weight); [masks glossary](../../../../glossary.md#term-boolean-mask); [ignore index glossary](../../../../glossary.md#ignore-index); [normalization glossary](../../../../glossary.md#term-normalization); [registry glossary](../../../../glossary.md#term-registry))</span>
- broadcasting, implicit expansion, implicit casts, nullable/default reduction, a default-mean
  overload, static facade, alias, primitive reduction spelling, or eager constants<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [mean glossary](../../../../glossary.md#term-mean); [facade glossary](../../../../glossary.md#term-facade); [alias glossary](../../../../glossary.md#aliasing-and-arena); [eager glossary](../../../../glossary.md#term-eager-evaluation); [constants glossary](../../../../glossary.md#term-constant))</span>
- value evaluation, storage reads, tolerances, numerical algorithm/pass/tree choice, gradients,
  adjoints, autograd traversal, compiler capture/validation/decomposition/fusion, backend support,
  lowering, kernels, prepare, runtime, publication, or execution<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication))</span>
- [parameter](../../../../glossary.md#term-parameter)/[module](../../../../glossary.md#term-module)/[optimizer](../../../../glossary.md#term-optimizer)/training-session/step/[checkpoint](../../../../glossary.md#term-checkpoint) [ownership](../../../../glossary.md#term-ownership-meanings) or mutation
- integral, [BOOL](../../../../glossary.md#data-type--datatype), [FLOAT16](../../../../glossary.md#term-float16), complex, sparse, quantized, or unsigned [prediction](../../../../glossary.md#term-prediction)/[target](../../../../glossary.md#term-target)/result data
- changes to existing arithmetic, reduction, Shape, promotion, Tensor factory, producer,
  provenance, compiler/runtime/training APIs, architecture, dependencies, Gradle, or another module<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Tensor factory glossary](../../../../glossary.md#tensor-factory); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- detailed specifications or implementation for tasks 0022A–0022B or 0023–0024

## Public and [operation](../../../../glossary.md#operation) contracts

### Kind, typed attributes, and [signature](../../../../glossary.md#term-signature)

`LossKind` contains exactly `MEAN_SQUARED_ERROR`. Its exact ordered signature is:

```java
OperationSignature.fixed(MeanSquaredErrorAttrs.class, 2, 1)
```

One [occurrence](../../../../glossary.md#term-occurrence) has ordered inputs `[prediction, target]` and ordered output `[loss]`. The receiver
is [prediction](../../../../glossary.md#term-prediction) at input position zero. There is no hidden input, saved output, sibling value, or
variable [cardinality](../../../../glossary.md#term-cardinality).

`LossReduction` is a public [immutable](../../../../glossary.md#term-immutable) enum vocabulary with exactly:

```java
NONE
SUM
MEAN
```

The enum does not itself implement [`OperationAttrs`](../../../../glossary.md#operationattrs); it is a typed configuration component reused
by family-specific immutable attributes. It carries no string parser, default, denominator,
[axis](../../../../glossary.md#term-axis), [mask](../../../../glossary.md#term-boolean-mask), ignore value, [Tensor](../../../../glossary.md#tensor), or executable behavior.

`MeanSquaredErrorAttrs` has exactly one component:

```java
LossReduction reduction
```

Its compact constructor null-checks the component with `NullPointerException("reduction")` and
retains the exact enum reference. The record contains no input, output [Shape](../../../../glossary.md#shape), denominator, [data
type](../../../../glossary.md#data-type--datatype), algorithm, [gradient](../../../../glossary.md#term-gradient), [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), or [runtime](../../../../glossary.md#term-runtime) state. No [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) is needed:
this [operation](../../../../glossary.md#operation) has no [scalar](../../../../glossary.md#term-scalar) semantic [parameter](../../../../glossary.md#term-parameter) beyond the typed reduction enum.

### Public receiver and [target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape)

The public surface is exactly:

```java
public Tensor meanSquaredError(Tensor target, LossReduction reduction)
```

[Prediction](../../../../glossary.md#term-prediction) and target must have equal [rank](../../../../glossary.md#term-rank) and positionally compatible [Dimensions](../../../../glossary.md#dimension). For each [axis](../../../../glossary.md#term-axis):

- structurally equal Dimensions pass;
- unequal static Dimensions fail locally;
- a non-structurally-equal pair involving an unresolved Dimension is accepted with an equality
  obligation for later compiler capture/binding; and<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>
- no [singleton](../../../../glossary.md#term-singleton), [scalar](../../../../glossary.md#term-scalar), leading-axis, right-aligned, or other [broadcasting](../../../../glossary.md#broadcasting) is accepted.

This exact-shape target policy prevents silent replication from changing the selected [`MEAN`](../../../../glossary.md#term-mean)
denominator. When `NONE` is selected, the result retains the exact prediction Shape reference.
The target Shape object is never substituted into the result. When `SUM` or `MEAN` is selected,
the result uses the shared `Shape.scalar()` reference.

Exact Shape failures are:

```text
meanSquaredError target rank must equal prediction rank: prediction=<predictionRank>, target=<targetRank>
meanSquaredError target dimension mismatch at axis <axis>: prediction=<predictionDimension>, target=<targetDimension>
```

### Formula, reduction, denominator, and empty domains

After converting both input values to the selected result type and computation format, the
unreduced [loss](../../../../glossary.md#loss) at logical [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `i` is:

```text
d_i = prediction_i - target_i
loss_i = d_i * d_i
```

Let `E` be the complete logical [element](../../../../glossary.md#term-element) count of the compatible [prediction](../../../../glossary.md#term-prediction)/[target](../../../../glossary.md#term-target) [Shape](../../../../glossary.md#shape) after all
unresolved [Dimensions](../../../../glossary.md#dimension) are bound:

```text
NONE: output_i = loss_i                         Shape = prediction Shape
SUM:  output   = sum_i(loss_i)                  Shape = scalar
MEAN: output   = sum_i(loss_i) / E              Shape = scalar
```

`E` includes every logical coordinate and is not a [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-only or leading-[axis](../../../../glossary.md#term-axis) count. A [scalar](../../../../glossary.md#term-scalar) has
`E = 1`. A Shape with any zero [extent](../../../../glossary.md#term-extent) has `E = 0`; no [elementwise](../../../../glossary.md#term-elementwise) subtraction or square is then
evaluated. `NONE` returns an empty result, `SUM` returns [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), and [`MEAN`](../../../../glossary.md#term-mean) returns [NaN](../../../../glossary.md#term-nan) for
the zero-over-zero empty mean. Construction does not need a static element count and does not
store `E` in attributes. Dynamic empty/non-empty selection is an execution-time consequence of
the bound Shape, not a [model](../../../../glossary.md#term-model)-construction branch.

For prediction `[1, 2, 4]` and target `[1, 4, 1]`, the unreduced losses are `[0, 4, 9]`, `SUM` is
`13`, and `MEAN` is `13 / 3`. This is a mathematical illustration; [Tensor](../../../../glossary.md#tensor) construction reads no
values. The explicit reduction vocabulary is comparable to official
[PyTorch MSE loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html)<span data-glossary-navigation="true"> ([MSE glossary](../../../../glossary.md#mean-squared-error--mse))</span>,
but this task deliberately selects no default, weighting, or broadcast behavior.

### Data types, result type, and accumulation

[Prediction](../../../../glossary.md#term-prediction) and [target](../../../../glossary.md#term-target) must each be [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64). Derive the result type exactly
once with:

```java
DataTypePromotion.promoteFloating(predictionType, targetType)
```

Both logical operands participate in arithmetic as that promoted type. Every reduction mode
returns that same result type; [`MEAN`](../../../../glossary.md#term-mean) does not widen the public result. No cast [producer](../../../../glossary.md#term-producer) or hidden
conversion [Tensor](../../../../glossary.md#tensor) is inserted.

BFLOAT16 and FLOAT32 results perform subtraction, multiplication, summation, and division in
FLOAT32. FLOAT64 results perform them in FLOAT64. Final values are rounded to the result format.
Equal-or-wider intermediates, stable summation, compensation, vectorization, parallelization,
[fused](../../../../glossary.md#specialization-and-fusion) operations, and [reassociation](../../../../glossary.md#term-reassociation) are allowed when later [conformance](../../../../glossary.md#term-conformance) tolerances and the exact
special-value classes below are preserved. Narrower computation, a fixed traversal/tree, bitwise
[identity](../../../../glossary.md#term-identity-meanings), [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign preservation, and identical [finite](../../../../glossary.md#term-finite) [rounding](../../../../glossary.md#term-rounding) across [backends](../../../../glossary.md#backend) are not
promised.

Exact type failures, checked in input order, are:

```text
meanSquaredError prediction must have a floating data type, but was <dataType>
meanSquaredError target must have a floating data type, but was <dataType>
```

### [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed zero](../../../../glossary.md#term-signed-zero), [overflow](../../../../glossary.md#term-overflow), [reassociation](../../../../glossary.md#term-reassociation), and determinism

Construction reads no values. For every evaluated [element](../../../../glossary.md#term-element) and reduction:

- a NaN [prediction](../../../../glossary.md#term-prediction) or [target](../../../../glossary.md#term-target) produces NaN squared error;
- equal-sign [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) subtract to NaN and therefore produce NaN squared error;
- one infinity and one finite value, or opposite-sign infinities, produce positive infinity after
  squaring;<span data-glossary-navigation="true"> ([positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- every exact zero difference, including results from any signed-zero pairing, squares to positive
  zero;<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- finite subtraction or squaring may overflow to positive infinity or underflow to positive zero
  in the selected computation format;<span data-glossary-navigation="true"> ([underflow glossary](../../../../glossary.md#term-underflow))</span>
- `SUM`/[`MEAN`](../../../../glossary.md#term-mean) propagates any NaN; otherwise any positive infinity makes the reduced result
  positive infinity; an all-finite exact-zero domain reduces to positive zero before mean
  division; and
- empty `SUM` is [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) while empty `MEAN` is NaN as specified above.

The abstract formula is non-negative for every non-NaN element. An implementation must not use an
algebraic expansion such as `p*p - 2*p*t + t*t` when it changes the required infinity/NaN classes.
Reduction traversal is not fixed, and permitted reassociation may change [finite](../../../../glossary.md#term-finite) [rounding](../../../../glossary.md#term-rounding) within
later [tolerance](../../../../glossary.md#term-tolerance). Separately constructed equal requests remain distinct [producers](../../../../glossary.md#term-producer); there is no
interning, [identity](../../../../glossary.md#term-identity-meanings) determinism, or bitwise-result guarantee.

### Result [metadata](../../../../glossary.md#term-metadata), [layout](../../../../glossary.md#layout), [gradients](../../../../glossary.md#term-gradient), [provenance](../../../../glossary.md#provenance), and identifiers

The result [descriptor](../../../../glossary.md#term-descriptor) uses the promoted result type, selected [Shape](../../../../glossary.md#shape), unresolved layout, and
`requiresGrad = prediction.requiresGrad() || target.requiresGrad()`. This eligibility is metadata
only and does not define or promise a gradient rule. The result is fresh, unlabeled, and
storage-free. Neither input nor its descriptor, label, storage, gradient flag, ID, Shape,
provenance, or layout is mutated.

Every success creates one `MeanSquaredErrorAttrs`, one [`Operation`](../../../../glossary.md#operation), one [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), one
output descriptor, one provenance value at output index zero, one [Tensor](../../../../glossary.md#tensor) wrapper, and one Tensor
ID. Provenance records exact ordered inputs `[prediction, target]`. No primitive arithmetic or
reduction [producers](../../../../glossary.md#term-producer) are decomposed beneath the [loss](../../../../glossary.md#loss) [occurrence](../../../../glossary.md#term-occurrence).

Every known local [validation](../../../../glossary.md#term-validation) failure occurs before factory delegation and consumes no Tensor ID,
[producer](../../../../glossary.md#term-producer), provenance, or wrapper. Factory identifier exhaustion retains the existing single-output
factory behavior; no [rollback](../../../../glossary.md#term-rollback) or new identifier policy is introduced.

### [Validation](../../../../glossary.md#term-validation) and construction order

`TensorLossExpressions.meanSquaredError(...)` performs exactly:

1. null-check [prediction](../../../../glossary.md#term-prediction), [target](../../../../glossary.md#term-target), then reduction;
2. validate prediction floating type, then target floating type;
3. require equal [rank](../../../../glossary.md#term-rank);
4. validate static/deferred [Dimension](../../../../glossary.md#dimension) compatibility in increasing logical-[axis](../../../../glossary.md#term-axis) order;
5. promote prediction and target types in [occurrence](../../../../glossary.md#term-occurrence) order;
6. construct `MeanSquaredErrorAttrs`;
7. select the exact prediction [Shape](../../../../glossary.md#shape) for `NONE` or shared `Shape.scalar()` for `SUM`/[`MEAN`](../../../../glossary.md#term-mean);
8. construct the unresolved result [descriptor](../../../../glossary.md#term-descriptor), [`Operation`](../../../../glossary.md#operation), and delegate exactly once to the
   single-output derived Tensor factory path.<span data-glossary-navigation="true"> ([Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>

Null messages are `prediction`, `target`, and `reduction`. The public receiver cannot be null at a
normal Java instance call, but the package-private helper retains explicit prediction validation
and tests lock it. The attributes constructor defensively repeats reduction null validation.
Existing promotion, descriptor, operation, [producer](../../../../glossary.md#term-producer), and identifier failures retain their current
messages.

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
- [Operation signatures](0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [Typed scalar contract](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- [Softmax semantics](0016i-softmax-semantic-kinds-and-attributes.md)<span data-glossary-navigation="true"> ([Softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- [Softmax Tensor expressions](0016j-softmax-tensor-expressions.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>

## Architecture constraints

- `modules/model` owns only backend-independent loss semantics, Tensor metadata, and immutable
  pre-capture provenance. Tensor remains public mutable API state and is not graph IR.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [loss glossary](../../../../glossary.md#loss); [Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [immutable glossary](../../../../glossary.md#term-immutable); [capture glossary](../../../../glossary.md#term-capture-meanings); [provenance glossary](../../../../glossary.md#provenance); [mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Operation types may consume foundational data-type values but must not import Tensor, compiler,
  training, runtime, prepare, engine, graph state, or backend types.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine))</span>
- Compiler owns graph capture, proof of deferred Shape equality, gradients, autograd/backward
  construction, decomposition, canonicalization, and optimization.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [Shape glossary](../../../../glossary.md#shape); [gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [decomposition glossary](../../../../glossary.md#term-decomposition); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- Backend prepare owns algorithm, lowering, specialization, fusion, kernel, and tolerance
  satisfaction. Runtime executes already-prepared work and must not consume <span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>[`Operation`](../../../../glossary.md#operation) in its hot
  path.<span data-glossary-navigation="true"> ([hot path glossary](../../../../glossary.md#term-hot-path))</span>
- The training extension may later consume loss outputs but does not own this model semantic
  operation. This task adds no session, optimizer, parameter, or state dependency.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [optimizer glossary](../../../../glossary.md#term-optimizer); [parameter glossary](../../../../glossary.md#term-parameter); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- No architecture, explanatory architecture, ADR, dependency, Gradle, cross-module, conformance,
  or integration change is authorized. Stop if the contract cannot be represented by the existing
  single-output producer and descriptor foundations.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [conformance glossary](../../../../glossary.md#term-conformance); [producer glossary](../../../../glossary.md#term-producer); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`

Package added:

- `io.github.pho001.synaptik.model.operation.loss` — cohesive public loss semantics and typed
  reduction/configuration values. It is distinct from aggregate reduction because a loss combines
  ordered prediction/target roles before optionally reducing.<span data-glossary-navigation="true"> ([aggregate reduction glossary](../../../../glossary.md#aggregate-reduction); [prediction glossary](../../../../glossary.md#term-prediction); [target glossary](../../../../glossary.md#term-target))</span>

Type placement:

- `...operation.loss.LossKind` — family-owned semantic [identity](../../../../glossary.md#term-identity-meanings) and [signature](../../../../glossary.md#term-signature).
- `...operation.loss.LossReduction` — shared public [loss](../../../../glossary.md#loss)-only reduction vocabulary.
- `...operation.loss.MeanSquaredErrorAttrs` — [immutable](../../../../glossary.md#term-immutable) [MSE](../../../../glossary.md#mean-squared-error--mse) reduction [parameters](../../../../glossary.md#term-parameter).
- `...tensor.TensorLossExpressions` — package-private validation, promotion, Shape, descriptor,
  and provenance construction owner, deliberately named for reuse by tasks 0022A–0022B without
  pre-implementing them.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [owner glossary](../../../../glossary.md#term-owner))</span>
- `...tensor.Tensor` — established public fluent receiver [facade](../../../../glossary.md#term-facade).

Tests mirror production packages where package-private helper access is needed.

## Affected files

Production (5):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/LossKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/LossReduction.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/loss/MeanSquaredErrorAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorLossExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (10):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/loss/LossSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`

The eight existing inventory tests change only their exact public [Tensor](../../../../glossary.md#tensor) method count from 185 to
186. They must not receive unrelated assertion or behavior changes.

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime) and Training APIs; related arithmetic, reduction,
[Shape](../../../../glossary.md#shape), promotion, [signature](../../../../glossary.md#term-signature), factory, [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), [normalization](../../../../glossary.md#term-normalization), and indexing contracts;
architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); dependencies; and other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 22 actual paths: five production, ten tests, and seven documentation/[planning](../../../../glossary.md#term-planning) paths. This
exceeds the planning guide's normal 18-path guardrail under the user's standing authorization
because the cohesive capability necessarily adds three public [loss](../../../../glossary.md#loss) semantic types, one bounded
construction helper, one [Tensor](../../../../glossary.md#tensor) method, two focused tests, updates all eight established exact
public-count inventories atomically, and completes the mandatory seven-path documentation [handoff](../../../../glossary.md#term-handoff).
Splitting those inventory or documentation updates would leave the public [API](../../../../glossary.md#term-application-programming-interface-api) locks broken or the
capability incomplete. Stop for path 23, another public method/type, another test/document, a
factory/foundation modification, any task-0022A+ implementation, cross-[module](../../../../glossary.md#term-module) work, architecture or
[Gradle](../../../../glossary.md#term-gradle) change, or unrelated cleanup.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document the kind, reduction enum, attributes, helper, and Tensor method: exact input
  roles/order, target Shape/no-broadcast policy, formulas, reductions, denominator, empty domains,
  types/computation, special values, metadata, IDs, failures, and lifecycle boundaries.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Every method/constructor input, non-void result, and expected failure has meaningful `@param`,
  `@return`, and `@throws` text under the [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc profile.
- [Tensor](../../../../glossary.md#tensor) API adds the exact [signature](../../../../glossary.md#term-signature), reduction/output table, `[1,2,4]` versus `[1,4,1]` example,
  no-broadcast/deferred-Shape rules, computation and special values, provenance/IDs, and current
  model versus planned compiler/backend/runtime/training boundaries.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Compile API records MSE as current model metadata only; capture, deferred equality proof,
  gradients, decomposition, validation, lowering, and execution remain planned.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [MSE glossary](../../../../glossary.md#mean-squared-error--mse); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [validation glossary](../../../../glossary.md#term-validation); [lowering glossary](../../../../glossary.md#lowering))</span>
- Review glossary terms loss, loss reduction, mean-squared error, prediction, target, and mean
  denominator; update only reusable distinctions.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [loss reduction glossary](../../../../glossary.md#loss-reduction); [mean-squared error glossary](../../../../glossary.md#mean-squared-error--mse); [prediction glossary](../../../../glossary.md#term-prediction); [mean glossary](../../../../glossary.md#term-mean))</span>
- Record reasoned no-change conclusions for Runtime and Training APIs, related contracts,
  architecture/ADRs/tests, conformance/integration, Gradle, dependencies, and other modules.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- `LossKind` has exactly one kind and exact `MeanSquaredErrorAttrs` fixed 2/1 signature; incompatible
  attributes and occurrence counts fail through existing contracts.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `LossReduction` has exactly `NONE`, `SUM`, [`MEAN`](../../../../glossary.md#term-mean); attrs retain the exact non-null value and no
  parser/default/options surface exists.
- Tensor exposes exactly one new public method with the signature in this task; public Tensor
  method count is exactly 186 and every existing inventory lock is synchronized.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Ordered inputs are exactly `[prediction, target]`; result provenance uses output index zero and
  one fresh producer/ID on success.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [producer glossary](../../../../glossary.md#term-producer))</span>
- BFLOAT16/FLOAT32/FLOAT64 pairs promote through current floating promotion; all other inputs fail
  in exact order and no cast is inserted.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- Equal ranks and exact/static-or-deferred positional Dimension compatibility are enforced; target
  broadcasting is absent.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [target glossary](../../../../glossary.md#term-target); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- `NONE` retains the exact [prediction](../../../../glossary.md#term-prediction) [Shape](../../../../glossary.md#shape); `SUM`/`MEAN` use `Shape.scalar()`; every layout is
  unresolved and gradient eligibility is the input logical OR.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Formula, accumulation, denominator, scalar/empty behavior, NaN/infinity/signed-zero/overflow,
  reassociation, and determinism match this task without construction-time value reads.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Null/type/rank/Dimension failures have exact messages and occur before factory delegation with no
  ID consumption.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- No gradient, compiler, backend, runtime, training-session, weights/mask/ignore/smoothing,
  categorical loss, registry, architecture, dependency, or Gradle behavior is added.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [weights glossary](../../../../glossary.md#term-weight); [mask glossary](../../../../glossary.md#term-boolean-mask); [loss glossary](../../../../glossary.md#loss); [registry glossary](../../../../glossary.md#term-registry); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- The exact 22-path maximum and package placement hold.
- A separate clean documentation-focused agent pass finalizes affected Javadocs, Tensor/Compile
  APIs, glossary impact, capabilities/task/master/roadmap records, and documentation validation in
  the same overall change.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [validation glossary](../../../../glossary.md#term-validation))</span>
- Task 0022 becomes `Complete` only after implementation, focused/final validation, clean docs
  handoff, evidence, completion summary, and synchronized status. Tasks 0022A–0022B and 0023–0024
  remain Draft without detailed specifications.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused validation while developing:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.loss.LossSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMeanSquaredErrorExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormInferenceExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorBinaryArithmeticTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLayerNormExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorLinearExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorMatmulExpressionTest \
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
whitespace checks; verifies the official PyTorch link and the numerical example; audits exact
22-path [scope](../../../../glossary.md#term-scope), package placement, exact public [signature](../../../../glossary.md#term-signature)/count, no detailed 0022A–0024 specs, and
status [synchronization](../../../../glossary.md#term-synchronization); and records reused Java evidence. It does not rerun successful Java tests
unless it changes executable Java behavior or records a concrete risk.

Repository-wide validation is deferred to the [model](../../../../glossary.md#term-model) capability [checkpoint](../../../../glossary.md#term-checkpoint) after task 0022B and [CI](../../../../glossary.md#term-continuous-integration-ci)
because this is a single-[module](../../../../glossary.md#term-module) [metadata](../../../../glossary.md#term-metadata)-construction task with no [dependency](../../../../glossary.md#term-dependency), architecture,
shared-build, or executable [backend](../../../../glossary.md#backend) change.

## Dependencies

- 0001–0002: floating types, promotion, Shape, Dimension, static/symbolic extents, rank, and scalar
  Shape.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [rank glossary](../../../../glossary.md#term-rank))</span>
- 0005–0007 and 0011–0013: operation, signature, descriptor, Tensor, factory, and provenance
  foundations.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor))</span>
- 0018K: exact attrs-class [signature](../../../../glossary.md#term-signature) and [occurrence](../../../../glossary.md#term-occurrence)-[cardinality](../../../../glossary.md#term-cardinality) [validation](../../../../glossary.md#term-validation).
- 0018L: unified [producer](../../../../glossary.md#term-producer) and output-index [provenance](../../../../glossary.md#provenance).
- 0018N: exact typed configuration precedent; this task confirms no [scalar](../../../../glossary.md#term-scalar) [parameter](../../../../glossary.md#term-parameter) is needed.
- 0018V: floating computation, reduction, empty-domain, special-value, and determinism precedent.

All dependencies are Complete.

## Follow-up tasks

- Draft task 0022A later owns dense-target categorical cross-entropy directly from logits with an
  explicit class axis and the shared loss reductions. It has no detailed specification yet.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [class axis glossary](../../../../glossary.md#class-axis); [loss glossary](../../../../glossary.md#loss))</span>
- Draft task 0022B later owns INT32/INT64 index-target logits cross-entropy, class-axis removal,
  optional ignore index, value-bound obligations, and the non-ignored <span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [axis glossary](../../../../glossary.md#term-axis); [ignore index glossary](../../../../glossary.md#ignore-index))</span>[`MEAN`](../../../../glossary.md#term-mean) denominator. It has no
  detailed specification yet.
- Task 0023 later owns selected compiler-generated backward semantic operations, not autograd
  traversal itself. This task defines no MSE gradient.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [MSE glossary](../../../../glossary.md#mean-squared-error--mse); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Compiler, prepare/runtime, concrete backends, and the training extension later own their existing
  capture, proof, autograd, lowering, execution, and coordination responsibilities.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backends glossary](../../../../glossary.md#backend); [capture glossary](../../../../glossary.md#term-capture-meanings); [lowering glossary](../../../../glossary.md#lowering))</span>
- Task 0024 remains the final [model](../../../../glossary.md#term-model) capability-selection audit.

## Architecture impact

Expected impact: None.

The task adds one [model](../../../../glossary.md#term-model)-owned pure semantic [operation](../../../../glossary.md#operation) and a focused operation-family package under
the existing model boundary. If implementation requires a new [dependency](../../../../glossary.md#term-dependency), [graph](../../../../glossary.md#graph)-local [identity](../../../../glossary.md#term-identity-meanings),
hidden state, [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend)/training type, [mutable](../../../../glossary.md#term-mutable) service, or architecture update, stop
and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0022-mean-squared-error-loss.md. Implement that task exactly as
specified. Do not implement categorical losses, gradients/autograd, compiler/backend/runtime/
training behavior, or later tasks. Stop and report any architecture, dependency, or maximum-scope
conflict. Do not commit or push unless separately authorized.

After Java implementation and recorded focused/final model-test evidence, hand the final diff and
evidence to a separate documentation-focused agent/thread with clean context. That pass must follow
docs/developer-guide/documentation-rules.md, inspect source/tests, finalize Javadocs, Tensor and
Compile APIs, glossary and planning records, run Javadoc/Markdown/scope/status/whitespace checks,
and reuse successful Java evidence unless executable Java changes.

Update this task's decisions, limitations, evidence, notes, completion summary, and status. Do not
mark Complete before the documentation pass and every acceptance criterion finishes.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Provide the clean documentation context with this task, the complete diff, focused/final Java
evidence, exact two-input/one-output roles, reduction/denominator/empty and special-value contracts,
[Shape](../../../../glossary.md#shape)/type/[layout](../../../../glossary.md#layout)/[gradient](../../../../glossary.md#term-gradient)/[provenance](../../../../glossary.md#provenance)/ID effects, architecture and 22-path limits, expected
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/glossary/[planning](../../../../glossary.md#term-planning) impact, official PyTorch link, and required [Javadoc](../../../../glossary.md#term-javadoc)/Markdown/
[scope](../../../../glossary.md#term-scope)/status evidence. It must inspect source/tests independently and record reasoned no-change
conclusions rather than relying on the handoff summary.

## Local decisions

- `TensorLossExpressions` rejects only unequal static/static Dimension pairs. Structural equality
  passes and every unequal pair involving an unresolved Dimension remains a later equality
  obligation; no broadcast representation is introduced.<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [Structural equality glossary](../../../../glossary.md#term-structural-equality))</span>
- `NONE` retains the exact [prediction](../../../../glossary.md#term-prediction) [Shape](../../../../glossary.md#shape) object, while both reduced forms reuse
  `Shape.scalar()`. This preserves the requested role asymmetry without storing Shape or element
  count in attributes.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- `LossReduction` remains [loss](../../../../glossary.md#loss)-only configuration rather than aggregate-reduction attributes.
  `MeanSquaredErrorAttrs` is the sole operation attributes type and retains only that exact enum
  value.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- The documentation example is explicitly mathematical because current model construction reads
  no values. It uses prediction <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>`[1, 2, 4]` and [target](../../../../glossary.md#term-target) `[1, 4, 1]`, yielding losses `[0, 4, 9]`,
  sum `13`, and [mean](../../../../glossary.md#term-mean) `13 / 3`.

## Known limitations

- Only exact-shape BFLOAT16/FLOAT32/FLOAT64 mean-squared error is current. There is no target
  broadcasting, weighting, mask, ignore index, smoothing, categorical loss, default reduction,
  or integral/BOOL loss domain.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [mean-squared error glossary](../../../../glossary.md#mean-squared-error--mse); [target glossary](../../../../glossary.md#term-target); [broadcasting glossary](../../../../glossary.md#broadcasting); [mask glossary](../../../../glossary.md#term-boolean-mask); [ignore index glossary](../../../../glossary.md#ignore-index); [loss glossary](../../../../glossary.md#loss); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- `requiresGrad` is eligibility metadata only. Gradient/adjoint construction, graph capture,
  deferred-Shape proof, decomposition, lowering, numerical execution, and training coordination
  remain planned in their owning layers.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [Gradient glossary](../../../../glossary.md#term-gradient); [adjoint glossary](../../../../glossary.md#term-adjoint); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering))</span>
- Numerical semantics permit reassociation and equal-or-wider intermediates subject to future
  conformance tolerances and the fixed special-value classes; traversal, bitwise identity, NaN
  payload/sign retention, and identical finite rounding are not promised.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [conformance glossary](../../../../glossary.md#term-conformance); [identity glossary](../../../../glossary.md#term-identity-meanings); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [finite glossary](../../../../glossary.md#term-finite); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Repository-wide validation remains deferred to the model loss-capability checkpoint after task
  0022B and CI, as specified for this single-module metadata task.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci); [module glossary](../../../../glossary.md#term-module))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0022_implementation` ran the prescribed focused Gradle
  selection containing <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`LossSemanticsTest`, `TensorMeanSquaredErrorExpressionTest`, `TensorTest`,
  and all eight exact-count inventory suites: `BUILD SUCCESSFUL`. It then ran
  `./gradlew :modules:model:test` exactly once after executable Java stabilized: `BUILD SUCCESSFUL
  in 1s`, three actionable tasks, one executed and two up-to-date. The documentation context
  reused this evidence because it changed Javadoc and Markdown only after that run.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [run glossary](../../../../glossary.md#term-run))</span>
- An earlier implementation-development `compileTestJava` failure came from a test lambda that
  captured a mutated local. The implementation context corrected the test before both successful
  runs above; it is not an unresolved product issue.
- Documentation context `/root/task_0022_implementation/task_0022_docs` applied the General,
  API/Javadoc, API-reference, Example, and Planning profiles. It independently reviewed the final
  five production paths, ten test paths, operation/signature, floating promotion, Shape/Dimension,
  descriptor/factory/producer/provenance foundations, Tensor/Compile/Runtime/Training API
  boundaries, glossary, capability baseline, master plan, roadmap, and this task.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `./gradlew :modules:model:javadoc`: `BUILD SUCCESSFUL in 2s`; two actionable tasks executed.
  This compiled the final Javadoc-only production edits and generated model Javadoc successfully.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- The targeted Ruby Markdown checker resolved 636 local links including 171 anchors across all
  seven affected documentation/planning files. Its first invocation failed before checking links
  because the installed Ruby lacks `Array#filter_map`; the compatible replacement then passed.
  Targeted checks also reported balanced Markdown fences, final newlines present, and no trailing
  whitespace.
- `curl -L --fail --silent --show-error --head` returned `HTTP/2 200` for the official PyTorch MSE
  reference. An independent arithmetic check produced <span data-glossary-navigation="true"> ([MSE glossary](../../../../glossary.md#mean-squared-error--mse))</span>`losses=[0,4,9] sum=13
  mean=4.33333333333`, matching `13 / 3`.
- The final scope audit found exactly 22 paths: five production, ten tests, and seven
  documentation/planning paths. <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`javap` found exactly 186 public declared Tensor methods and the
  sole exact receiver signature
  `Tensor meanSquaredError(Tensor, LossReduction)`. It also confirmed package-private, final,
  field-free `TensorLossExpressions` with exactly one package entry and its two private validators.
  Source/package checks confirmed the three loss types under <span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss))</span>`operation.loss`, exact enum
  vocabularies, fixed 2-input/1-output signature, and no detailed task files for 0022A, 0022B,
  0023, or 0024.
- Final status audit synchronized task 0022 to `Complete` in this task, the model master plan, and
  roadmap. Tasks 0022A–0022B and 0023–0024 remain `Draft`; no later detailed specification exists.
- Final targeted Markdown checks and `git diff --check` passed after status/evidence
  synchronization.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>

## Implementation notes

- Added exactly `LossKind.MEAN_SQUARED_ERROR`, `LossReduction.NONE/SUM/MEAN`, and [immutable](../../../../glossary.md#term-immutable)
  `MeanSquaredErrorAttrs(reduction)` with the fixed exact two-input/one-output [signature](../../../../glossary.md#term-signature).
- Added field-free package-private `TensorLossExpressions` and exactly one public receiver method.
  Construction validates nulls, floating types, rank, and positional static Dimensions before
  promotion and factory delegation; records ordered <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Dimensions glossary](../../../../glossary.md#dimension))</span>`[prediction, target]`; selects exact
  none/scalar Shape metadata; combines gradient eligibility by logical OR; and creates one fresh
  producer, ID, and output-index-zero provenance per success.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Added focused semantic and expression tests and changed only the expected public Tensor count
  from 185 to 186 in the eight existing inventory tests.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Finalized Javadocs in all five authorized production paths and added the Tensor API expression
  and semantic references, Compile API model/planned boundary, reusable glossary definitions, and
  synchronized capability/task/master/roadmap records.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [model glossary](../../../../glossary.md#term-model))</span>
- Runtime API required no change because no prepared execution or run behavior exists. Training
  API required no change because the loss is model metadata and adds no session, parameter,
  optimizer, step, or gradient contract.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [run glossary](../../../../glossary.md#term-run); [loss glossary](../../../../glossary.md#loss); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Existing arithmetic/reduction, Shape/promotion, signature/factory, producer/provenance,
  normalization/indexing, Runtime/Training, architecture/ADR/test, conformance/integration,
  Gradle/dependency, and other-module contracts remain accurate: the implementation composes
  their existing boundaries without changing them. Compile API changed only to expose the new
  current model metadata and explicitly preserve planned compiler responsibilities.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [compiler glossary](../../../../glossary.md#term-compiler))</span>

## Completion summary

- Completed changes: added the selected exact-shape MSE semantic vocabulary, attributes,
  construction helper, sole Tensor receiver, tests, Javadocs, API references, glossary terms, and
  synchronized planning records.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [MSE glossary](../../../../glossary.md#mean-squared-error--mse); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly five production, ten test, and seven documentation/planning
  paths; 22 total.
- Tests and validation: reused successful prescribed focused and one final model-test run;
  independently passed model Javadoc, Markdown links/anchors/fences/newlines/whitespace, official-
  reference, numerical-example, exact-scope/package/surface/status, and final diff checks.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: completed in clean context
  `/root/task_0022_implementation/task_0022_docs` under the selected documentation profiles.
- Documentation impact: Tensor and Compile API references and planning records now distinguish
  current MSE metadata from planned compiler/backend/runtime/training behavior.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: finalized all five affected production paths; no executable Java changed during
  the documentation pass.
- Glossary impact: added reusable loss, loss-reduction, prediction/target, and MSE/mean-denominator
  distinctions and integrated the new kind, attributes, and provenance into existing entries.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [prediction glossary](../../../../glossary.md#term-prediction); [target glossary](../../../../glossary.md#term-target); [mean glossary](../../../../glossary.md#term-mean); [provenance glossary](../../../../glossary.md#provenance))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0022. Draft tasks 0022A–0022B retain categorical-loss work;
  later compiler, backend, runtime, training, and checkpoint work remains separately planned.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

Status: Complete
