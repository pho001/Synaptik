# Task 0021C: [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[Normalization](../../../../glossary.md#term-normalization) Training and [Statistic](../../../../glossary.md#term-statistic) Transition

## Status

Complete

## Goal

Add one first-class [backend](../../../../glossary.md#backend)-independent `BATCH_NORM_TRAINING` meaning and one public pure [Tensor](../../../../glossary.md#tensor)
expression. One [occurrence](../../../../glossary.md#term-occurrence) computes current-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) per-[channel](../../../../glossary.md#term-channel) statistics, normalized affine output,
an explicit next running [mean](../../../../glossary.md#term-mean) and [variance](../../../../glossary.md#term-variance), and the saved batch values needed by later [compiler](../../../../glossary.md#term-compiler)
[autograd](../../../../glossary.md#autograd). All old and next statistics are visible [graph](../../../../glossary.md#graph) values; construction owns no state across
calls, [training session](../../../../glossary.md#training-session--trainingsession), [checkpoint](../../../../glossary.md#term-checkpoint), mutation, or hidden training/[evaluation](../../../../glossary.md#term-evaluation) mode.

[Mental model](../../../../glossary.md#term-mental-model):

```text
data + affine + old running statistics
  -> one pure five-output producer
     -> public output + public next running statistics
     -> hidden saved mean + inverse standard deviation for compiler autograd
```

## [Scope](../../../../glossary.md#term-scope)

- Append `BatchNormKind.BATCH_NORM_TRAINING` without changing completed [inference](../../../../glossary.md#term-inference) semantics.
- Add [immutable](../../../../glossary.md#term-immutable) public `BatchNormTrainingAttrs(channelAxis, momentum, epsilon)`.
- Give the training kind exactly one fixed five-input/five-output [signature](../../../../glossary.md#term-signature).
- Add exactly this receiver method and public result [carrier](../../../../glossary.md#term-storage-carrier):

  ```java
  public BatchNormTrainingResult batchNormTraining(
          int channelAxis,
          Tensor scale,
          Tensor bias,
          Tensor runningMean,
          Tensor runningVariance,
          ScalarValue momentum,
          ScalarValue epsilon)

  public record BatchNormTrainingResult(
          Tensor output,
          Tensor nextRunningMean,
          Tensor nextRunningVariance) {}
  ```

- Add one package-private field-free `TensorBatchNormTrainingExpressions` helper.
- Record exact ordered inputs and outputs, Shape/type metadata, gradient eligibility, shared
  producer positions, identifier effects, failure atomicity, and validation messages.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [producer glossary](../../../../glossary.md#term-producer); [failure atomicity glossary](../../../../glossary.md#term-failure-atomicity); [validation glossary](../../../../glossary.md#term-validation))</span>
- Select the reduction domain, biased forward variance, unbiased running-variance estimate,
  new-batch-weight momentum convention, epsilon placement, saved inverse standard deviation,
  computation formats, and special-value contract.<span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [weight glossary](../../../../glossary.md#term-weight); [momentum glossary](../../../../glossary.md#term-momentum); [epsilon glossary](../../../../glossary.md#epsilon); [standard deviation glossary](../../../../glossary.md#term-standard-deviation))</span>
- Finalize Javadocs, Tensor and Compile API references, glossary, capabilities, task evidence,
  master plan, and roadmap in the mandatory clean documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- retaining, transporting, assigning, or mutating running statistics across calls; parameter,
  buffer, module, optimizer, training-session, loop, step, checkpoint, or serialization ownership<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [module glossary](../../../../glossary.md#term-module); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- a combined inference/training kind, training/evaluation flag, mutable layer, global mode,
  service locator, registry, hidden state, in-place update, or stateful convenience<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [evaluation glossary](../../../../glossary.md#term-evaluation); [mutable glossary](../../../../glossary.md#term-mutable); [service locator glossary](../../../../glossary.md#term-service-locator); [registry glossary](../../../../glossary.md#term-registry))</span>
- optional affine or running-statistic inputs, defaults, nullable operands, implicit constants,
  cumulative-average mode, batch counter, optional momentum, or attrs/options-taking public entry<span data-glossary-navigation="true"> ([statistic glossary](../../../../glossary.md#term-statistic); [constants glossary](../../../../glossary.md#term-constant); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [momentum glossary](../../../../glossary.md#term-momentum))</span>
- public saved mean, saved variance, saved inverse-standard-deviation, sibling-output lookup, or a
  public carrier component beyond the three selected public results<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [lookup glossary](../../../../glossary.md#term-lookup); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- compiler capture, deferred-constraint representation/proof, saved-value lifetime or
  materialization, gradient formulas, autograd traversal, backward construction, publication,
  liveness, memory planning, or graph optimization<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [materialization glossary](../../../../glossary.md#materialization); [gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [liveness glossary](../../../../glossary.md#term-liveness); [planning glossary](../../../../glossary.md#term-planning); [graph optimization glossary](../../../../glossary.md#forward-graph-optimization))</span>
- runtime execution/publication, cross-run storage, prepared state, session scheduling, backend
  support, lowering, fusion, algorithm/pass/tree/traversal choice, kernels, or tolerances<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [kernels glossary](../../../../glossary.md#kernel))</span>
- integral, BOOL, FLOAT16, complex, sparse, quantized, or unsigned data; cast insertion; configurable
  result, accumulator, or computation type<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16); [accumulator glossary](../../../../glossary.md#accumulator-type))</span>
- changes to inference, layer/RMS normalization, reductions, typed scalars, Shape, producer/factory
  foundations, architecture, dependencies, Gradle, another module, or detailed tasks 0022–0024<span data-glossary-navigation="true"> ([RMS normalization glossary](../../../../glossary.md#rms-normalization); [scalars glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape); [producer glossary](../../../../glossary.md#term-producer); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Public and [operation](../../../../glossary.md#operation) contracts

### Kind, attributes, and [signature](../../../../glossary.md#term-signature)

`BatchNormKind` retains `BATCH_NORM_INFERENCE` first and appends `BATCH_NORM_TRAINING`. The new
kind accepts only:

| Attributes | Ordered inputs | Ordered outputs |
|---|---|---|
| `BatchNormTrainingAttrs` | `[input, scale, bias, runningMean, runningVariance]` | `[output, nextRunningMean, nextRunningVariance, savedBatchMean, savedInverseStandardDeviation]` |

Its signature is exactly:

```java
OperationSignature.fixed(BatchNormTrainingAttrs.class, 5, 5)
```

`BatchNormTrainingAttrs` has exactly these components, in order:

```java
int channelAxis
ScalarValue momentum
ScalarValue epsilon
```

The compact constructor validates component order: non-negative normalized [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis), non-null
floating [momentum](../../../../glossary.md#term-momentum), [finite](../../../../glossary.md#term-finite) momentum in the closed interval `[0, 1]`, non-null floating [epsilon](../../../../glossary.md#epsilon),
then finite strictly positive epsilon. It retains both exact [immutable](../../../../glossary.md#term-immutable) [scalar](../../../../glossary.md#term-scalar) references. Exact
intrinsic failures are:

```text
channelAxis must be non-negative: <channelAxis>
momentum
momentum must have a floating data type, but was <dataType>
momentum must be finite and in [0, 1]: <momentum>
epsilon
epsilon must have a floating data type, but was <dataType>
epsilon must be finite and positive: <epsilon>
```

Both signed momentum zeros are accepted and represent zero new-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [weight](../../../../glossary.md#term-weight). Positive/[negative
infinity](../../../../glossary.md#nan-infinity-and-signed-zero), [NaN](../../../../glossary.md#term-nan), and finite values outside `[0, 1]` fail. Epsilon rejects both signed zeros,
non-finite values, and negatives exactly as completed [inference](../../../../glossary.md#term-inference) attrs do.

### Public receiver, result, and output visibility

The receiver is [producer](../../../../glossary.md#term-producer) input zero. There is no static method, [alias](../../../../glossary.md#aliasing-and-arena), overload, attrs-taking
entry, or training flag. `BatchNormTrainingResult` components are exactly `output`,
`nextRunningMean`, and `nextRunningVariance`; its compact constructor null-checks them in that
order with [parameter](../../../../glossary.md#term-parameter)-name messages, retains exact references, and performs no producer discovery.

Factory output slots zero through four are all real [Tensor](../../../../glossary.md#tensor) outputs sharing one producer. Slots
zero, one, and two are returned by the result [carrier](../../../../glossary.md#term-storage-carrier). The helper intentionally discards the
wrappers for slots three and four after construction; their exact [descriptors](../../../../glossary.md#term-descriptor) and indexed output
positions remain reachable through the shared producer from each public result. Later [compiler](../../../../glossary.md#term-compiler)
[capture](../../../../glossary.md#term-capture-meanings) may create [graph](../../../../glossary.md#graph) values for those slots and own their saved-value [lifetime](../../../../glossary.md#term-lifetime). [Model](../../../../glossary.md#term-model) adds no
public accessor or retention policy for them.

Saved inverse [standard deviation](../../../../glossary.md#term-standard-deviation) is selected instead of saved [variance](../../../../glossary.md#term-variance) because backward
standardization consumes the denominator reciprocal directly. This task defines only its forward
value and descriptor, not a [gradient](../../../../glossary.md#term-gradient) formula or compiler lifetime.

### Input [rank](../../../../glossary.md#term-rank), [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis), [vector](../../../../glossary.md#term-vector) [Shapes](../../../../glossary.md#shape), and [layout](../../../../glossary.md#layout)

Input rank `R` must be at least two. Normalize the caller axis exactly once through
`Shape.normalizeAxis`; valid raw values are `[-R, R - 1]`. Every logical axis may be the channel
axis. No [NCHW](../../../../glossary.md#term-nchw), [NHWC](../../../../glossary.md#term-nhwc), [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-axis position, physical layout, [stride](../../../../glossary.md#term-stride), or resolved-layout meaning is
inferred.

Scale, [bias](../../../../glossary.md#term-bias), old running [mean](../../../../glossary.md#term-mean), and old running [variance](../../../../glossary.md#term-variance) are mandatory exact [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) vectors.
Their sole [extent](../../../../glossary.md#term-extent) must equal the input channel [Dimension](../../../../glossary.md#dimension) `C`:

- structurally equal Dimension values pass;
- unequal static values fail locally;
- unequal pairs involving an unresolved Dimension defer equality to [compiler](../../../../glossary.md#term-compiler) [binding](../../../../glossary.md#term-binding); and
- no [broadcasting](../../../../glossary.md#broadcasting), expansion, [scalar](../../../../glossary.md#term-scalar), `[1,C]`, or full-rank [singleton](../../../../glossary.md#term-singleton) form is accepted.

The normalized output retains the exact input Shape reference. Construct one rank-one [statistic](../../../../glossary.md#term-statistic)
Shape from the exact input channel Dimension; next running mean, next running variance, saved batch
mean, and saved inverse [standard deviation](../../../../glossary.md#term-standard-deviation) share that exact Shape reference. All five output
layouts are unresolved regardless of input layouts.

### Reduction [axes](../../../../glossary.md#term-axis), count, and empty domains

For each [channel](../../../../glossary.md#term-channel) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `c`, reduce over every input axis except `channelAxis`, in logical axis
order. There is no separately distinguished [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) axis. Let `N` be the mathematical product of all
non-channel extents. Because [rank](../../../../glossary.md#term-rank) is at least two, the reduction-axis list is non-empty.

The unbiased running-[variance](../../../../glossary.md#term-variance) estimate requires `N > 1` whenever `C > 0`:

- static `C == 0` is valid for every non-negative reduction Shape; all five outputs are empty and
  no mean, variance, inverse standard deviation, affine value, or running update is evaluated;<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [mean glossary](../../../../glossary.md#term-mean); [standard deviation glossary](../../../../glossary.md#term-standard-deviation))</span>
- static `C > 0` with statically known `N` equal to zero or one fails locally;
- static or bound `C > 0` requires `N >= 2`;
- unresolved cases defer the exact obligation `C == 0 || N >= 2`; and
- a zero on a non-channel axis therefore fails for a statically positive channel extent, rather
  than producing NaN or a sentinel statistic.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [NaN glossary](../../../../glossary.md#term-nan); [statistic glossary](../../../../glossary.md#term-statistic))</span>

Determine static validity using an early zero check and checked multiplication as in statistical
reduction precedent. A non-zero product overflowing `long` is necessarily at least two and is
valid; `N` is not stored in attrs. Exact failure is:

```text
batchNormTraining reduction domain count <count> must be at least 2 when channel extent is non-zero
```

### Formula, estimates, saved values, and transition

For [channel](../../../../glossary.md#term-channel) `c`, with reduction-domain values `x_i`:

```text
batchMean[c]             = sum_i(x_i) / N
sumSquaredDeviation[c]   = sum_i((x_i - batchMean[c])^2)
biasedBatchVariance[c]   = sumSquaredDeviation[c] / N
unbiasedBatchVariance[c] = sumSquaredDeviation[c] / (N - 1)
savedInvStd[c]           = 1 / sqrt(biasedBatchVariance[c] + epsilon)
output_i                 = ((x_i - batchMean[c]) * savedInvStd[c]) * scale[c] + bias[c]
nextRunningMean[c]       = (1 - momentum) * runningMean[c]
                           + momentum * batchMean[c]
nextRunningVariance[c]   = (1 - momentum) * runningVariance[c]
                           + momentum * unbiasedBatchVariance[c]
```

The forward normalized output uses biased [population variance](../../../../glossary.md#population-variance). The running-[variance](../../../../glossary.md#term-variance) transition
uses the corresponding correction-one unbiased estimate. [Epsilon](../../../../glossary.md#epsilon) is used only inside the forward
inverse-standard-deviation [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); it does not alter either variance output or running update.
[Momentum](../../../../glossary.md#term-momentum) is the [weight](../../../../glossary.md#term-weight) of the new [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), not the weight of the old [statistic](../../../../glossary.md#term-statistic). No update is applied
in place.

For one [FLOAT64](../../../../glossary.md#term-float64) channel containing `[1, 2, 3]`, scale `2`, [bias](../../../../glossary.md#term-bias) `0.5`, old [mean](../../../../glossary.md#term-mean) `10`, old variance
`4`, momentum `0.25`, and epsilon `1e-5`:

```text
batch mean                  = 2
biased batch variance       = 2 / 3
unbiased batch variance     = 1
saved inverse std           = 1 / sqrt(2 / 3 + 1e-5)
next running mean           = 8
next running variance       = 3.25
normalized output           ~= [-1.949471, 0.5, 2.949471]
```

This is a mathematical illustration; construction reads no values. The biased-forward/unbiased-
running distinction and selected new-batch-weight convention align with official
[PyTorch BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html).
Official [ONNX BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html)<span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx))</span>
also supplies a pure explicit running-stat transition and population forward formula, but this
task deliberately does not copy its combined mode, fixed channel position, default [scalars](../../../../glossary.md#term-scalar),
three-output surface, or population running-variance update.

### Data types and computation formats

All five [Tensor](../../../../glossary.md#tensor) inputs must be [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64). Derive one [occurrence](../../../../glossary.md#term-occurrence) result type by
`DataTypePromotion.promoteFloating` in input order: input with scale, then [bias](../../../../glossary.md#term-bias), running [mean](../../../../glossary.md#term-mean), and
running [variance](../../../../glossary.md#term-variance). Every output [descriptor](../../../../glossary.md#term-descriptor) uses that result type. [Momentum](../../../../glossary.md#term-momentum) and [epsilon](../../../../glossary.md#epsilon) must each
have exactly that type; no [scalar](../../../../glossary.md#term-scalar) promotion, cast [producer](../../../../glossary.md#term-producer), or hidden conversion is inserted.

BFLOAT16 and FLOAT32 results compute [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) reductions, standardization, affine arithmetic, and
running updates in FLOAT32. FLOAT64 results compute in FLOAT64. Final outputs are rounded to the
result format. Equal-or-wider intermediate arithmetic, stable algorithms, compensation,
parallelization, and [reassociation](../../../../glossary.md#term-reassociation) are allowed when future [tolerance](../../../../glossary.md#term-tolerance) and the exact special-value
classes below are preserved. Narrower accumulation, a fixed pass count/tree, bitwise equality,
[NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign preservation, and cross-[backend](../../../../glossary.md#backend) identical [finite](../../../../glossary.md#term-finite) [rounding](../../../../glossary.md#term-rounding) are not promised.

Exact public-boundary type failures are:

```text
batchNormTraining <role> must have a floating data type, but was <dataType>
batchNormTraining momentum data type must match result data type: momentum=<momentumType>, result=<resultType>
batchNormTraining epsilon data type must match result data type: epsilon=<epsilonType>, result=<resultType>
```

`<role>` is checked in order: input, scale, bias, runningMean, runningVariance.

### Old [variance](../../../../glossary.md#term-variance), [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed zero](../../../../glossary.md#term-signed-zero), [overflow](../../../../glossary.md#term-overflow), and determinism

Construction reads no values. Old running variance may be negative and is not clamped, repaired,
or rejected. It does not participate in normalized output or saved statistics; it participates
only in `nextRunningVariance`. A [finite](../../../../glossary.md#term-finite) negative old value may therefore produce a negative next
value. [Negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) and NaN follow ordinary weighted-transition arithmetic.

- Any NaN or infinity in a non-empty batch domain makes batch mean, both batch-variance estimates,
  saved inverse standard deviation, and normalized outputs for that channel NaN; old statistics do
  not repair them.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [mean glossary](../../../../glossary.md#term-mean); [standard deviation glossary](../../../../glossary.md#term-standard-deviation); [channel glossary](../../../../glossary.md#term-channel))</span>
- A valid finite constant domain, including signed zeros, has positive-zero biased and unbiased
  variances and positive saved inverse standard deviation. Batch-mean zero sign follows the
  selected floating SUM/MEAN rule. Centering, affine operations, and transitions otherwise follow
  ordinary floating signed-zero arithmetic; no final zero sign is canonicalized.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- NaN/infinity in scale or bias follows ordinary affine arithmetic. NaN/infinity in an old running
  statistic affects only its corresponding next statistic unless batch arithmetic is already NaN.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [running statistic glossary](../../../../glossary.md#term-running-statistic); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- Momentum endpoints use the written floating formula. A zero coefficient does not suppress NaN
  or infinity (<span data-glossary-navigation="true"> ([Momentum glossary](../../../../glossary.md#term-momentum))</span>`0 * NaN` and `0 * infinity` are NaN); finite momentum zero preserves finite old
  statistics, and finite momentum one selects finite batch estimates.
- Finite centered, affine, or transition arithmetic may overflow or underflow in the selected
  computation format and follows ordinary infinity and signed-zero behavior. Variance remains the
  non-negative exact squared-deviation target; an unstable cancellation formula that invents a
  negative finite variance is not a conforming semantic result.<span data-glossary-navigation="true"> ([underflow glossary](../../../../glossary.md#term-underflow); [target glossary](../../../../glossary.md#term-target); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- Reduction traversal is not fixed. Axis order cannot change the abstract result; conforming
  reassociation may change finite rounding within later tolerance. Equal separately constructed
  requests remain distinct producers, so bitwise or identity determinism is not promised.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([Axis glossary](../../../../glossary.md#term-axis); [reassociation glossary](../../../../glossary.md#term-reassociation); [rounding glossary](../../../../glossary.md#term-rounding); [tolerance glossary](../../../../glossary.md#term-tolerance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

### Result [metadata](../../../../glossary.md#term-metadata), [gradients](../../../../glossary.md#term-gradient), [provenance](../../../../glossary.md#provenance), and identifiers

All outputs are fresh, unlabeled, storage-free, and [layout](../../../../glossary.md#layout)-unresolved. Their gradient-eligibility
metadata reflects direct mathematical dependencies:

| Slot | Output | [Shape](../../../../glossary.md#shape) | `requiresGrad` |
|---|---|---|---|
| 0 | normalized output | exact input Shape | `input || scale || [bias](../../../../glossary.md#term-bias)` |
| 1 | next running [mean](../../../../glossary.md#term-mean) | shared `[C]` Shape | `input || runningMean` |
| 2 | next running [variance](../../../../glossary.md#term-variance) | shared `[C]` Shape | `input || runningVariance` |
| 3 | saved [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) mean | shared `[C]` Shape | `input` |
| 4 | saved inverse [standard deviation](../../../../glossary.md#term-standard-deviation) | shared `[C]` Shape | `input` |

These flags are metadata only and do not define gradients. Old running statistics do not affect
slot zero. Scale and bias do not affect [statistic](../../../../glossary.md#term-statistic) outputs.

Each success creates one attrs value, one [Operation](../../../../glossary.md#operation), one [TensorProducer](../../../../glossary.md#producer--tensorproducer), five [descriptors](../../../../glossary.md#term-descriptor), five
provenance values, five [Tensor](../../../../glossary.md#tensor) wrappers and IDs in output order, and one public result record. All
provenance values share the exact [producer](../../../../glossary.md#term-producer) and use indices zero through four. Public slots zero,
one, and two retain the producer; slots three and four remain producer-described after their local
wrappers are discarded. No input, descriptor, label, storage, gradient flag, ID, or provenance is
mutated.

Known local failures occur before factory delegation and consume no Tensor ID. Factory identifier
exhaustion follows the existing multi-output contract: IDs for earlier positions may remain
consumed, no partial list or result is returned, and there is no [rollback](../../../../glossary.md#term-rollback). This allocation effect
is not a running-statistic update.

### [Validation](../../../../glossary.md#term-validation) and construction order

`TensorBatchNormTrainingExpressions.apply(...)` performs exactly:

1. null-check input, scale, [bias](../../../../glossary.md#term-bias), running [mean](../../../../glossary.md#term-mean), running [variance](../../../../glossary.md#term-variance), [momentum](../../../../glossary.md#term-momentum), then [epsilon](../../../../glossary.md#epsilon);
2. validate [Tensor](../../../../glossary.md#tensor) floating types in input order;
3. require input [rank](../../../../glossary.md#term-rank) at least two;
4. normalize [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis) exactly once;
5. require rank one for scale, bias, running mean, then running variance;
6. validate static/deferred channel compatibility in the same role order;
7. validate the static reduction-domain obligation;
8. promote five Tensor types in [occurrence](../../../../glossary.md#term-occurrence) order;
9. validate momentum exact result type, then epsilon exact result type;
10. construct attrs, the exact output/statistic Shapes and descriptors, Operation, and delegate
    exactly once to <span data-glossary-navigation="true"> ([statistic glossary](../../../../glossary.md#term-statistic); [descriptors glossary](../../../../glossary.md#term-descriptor); [Operation glossary](../../../../glossary.md#operation))</span>`TensorFactory.createDerivedOutputs(...)`;
11. read all five slots by position, intentionally discard saved wrappers, and construct the public
    result from slots zero through two.

Input rank and [vector](../../../../glossary.md#term-vector) diagnostics are:

```text
batchNormTraining input rank must be at least 2, but was <rank>
batchNormTraining <role> rank must be one, but was <rank>
batchNormTraining <role> channel dimension mismatch: input=<inputDimension>, <role>=<roleDimension>
```

Invalid raw axes use the current [Shape](../../../../glossary.md#shape) diagnostic. Attributes defensively repeat intrinsic axis
and [scalar](../../../../glossary.md#term-scalar) validation. Result-[carrier](../../../../glossary.md#term-storage-carrier) construction cannot fail for factory-produced non-null
outputs; an unexpected failure is an implementation defect and does not justify [rollback](../../../../glossary.md#term-rollback).

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Runtime/prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Capabilities](../capabilities.md)
- [Master plan](../master-plan.md)
- [Shared multi-output provenance](0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- [Typed scalar contract](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Statistical reductions](0018v-multi-axis-and-statistical-reductions.md)
- [Explicit state foundation](0019b-explicit-graph-rng-state-foundation.md)
- [Explicit dropout transition](0019b1-explicit-graph-dropout-construction.md)<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- [Batch-normalization inference](0021b-batch-normalization-inference.md)<span data-glossary-navigation="true"> ([Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [inference glossary](../../../../glossary.md#term-inference))</span>

## Architecture constraints

- `modules/model` owns only pure backend-independent operation semantics, Tensor metadata, and
  immutable pre-capture provenance. Tensor remains public mutable API state, not IR.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation); [Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance); [mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- [Compiler](../../../../glossary.md#term-compiler) owns [capture](../../../../glossary.md#term-capture-meanings), deferred proof, [autograd](../../../../glossary.md#autograd)/backward construction, and saved-value [lifetime](../../../../glossary.md#term-lifetime).
- Runtime owns prepared execution/publication and per-run state. Concrete backends own algorithms,
  lowering, kernels, and numerical tolerance satisfaction.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [publication glossary](../../../../glossary.md#term-publication); [run state glossary](../../../../glossary.md#run-state--runstate); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- `extensions/training` may later own cross-step/session coordination and checkpoint policy; this
  task must not transport or retain next statistics after returning them.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- No architecture, dependency, Gradle, compiler, runtime, backend, or extension change is
  authorized. Stop if the exact five-output producer cannot represent the contract.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [producer glossary](../../../../glossary.md#term-producer))</span>

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.normalization`
- `io.github.pho001.synaptik.model.tensor`
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`

Packages added: None.

Type placement:

- `...operation.normalization.BatchNormKind` — existing [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) semantic family.
- `...operation.normalization.BatchNormTrainingAttrs` — public immutable training occurrence
  parameters.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [occurrence glossary](../../../../glossary.md#term-occurrence); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- `...tensor.BatchNormTrainingResult` — small public [carrier](../../../../glossary.md#term-storage-carrier) for the three caller-visible outputs.
- `...tensor.TensorBatchNormTrainingExpressions` — package-private validation and shared-output
  construction owner.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [owner glossary](../../../../glossary.md#term-owner))</span>
- `...tensor.Tensor` — established fluent receiver [facade](../../../../glossary.md#term-facade).

Tests mirror production packages for package-private access.

## Affected files

Production (5):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/BatchNormKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/BatchNormTrainingAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/BatchNormTrainingResult.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormTrainingExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (11):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/normalization/BatchNormTrainingSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/normalization/BatchNormInferenceSemanticsTest.java`
  — required existing enum-inventory lock; appending the training kind preserves its fixed
  inference five-input/one-output assertions while expanding the exact family inventory.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormTrainingExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`

The eight existing inventory tests change only the exact public [Tensor](../../../../glossary.md#tensor) count from 184 to 185,
except the focused [inference](../../../../glossary.md#term-inference) test may also lock that its completed one-output contract is
unchanged.

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime) and Training APIs; related [normalization](../../../../glossary.md#term-normalization), reduction,
[scalar](../../../../glossary.md#term-scalar), [Shape](../../../../glossary.md#shape), [signature](../../../../glossary.md#term-signature), factory/[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) contracts; architecture/ADRs/tests;
[conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other [modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 23 actual paths: five production, eleven tests, and seven documentation/[planning](../../../../glossary.md#term-planning) paths.
This exceeds the planning guide's normal 18-path guardrail under the user's standing authorization
because one cohesive public multi-output capability necessarily adds kind/attrs/result/helper/
[Tensor](../../../../glossary.md#tensor) production contracts, two focused tests, updates all eight established exact public-count
assertions together, and completes the mandatory seven-path documentation [handoff](../../../../glossary.md#term-handoff). The 22-path
plan omitted one existing [inference](../../../../glossary.md#term-inference) semantics test that locked the complete `BatchNormKind`
inventory. Appending the required training kind made its inventory update unavoidable; this is the
sole authorized path-23 exception and does not alter inference's fixed 5/1 contract. Splitting the
change would leave a broken inventory or an incomplete public semantic capability. Stop for path
24, another public method or type, factory/foundation modification, another document/test,
cross-[module](../../../../glossary.md#term-module) work, architecture/[Gradle](../../../../glossary.md#term-gradle) change, or a detailed 0022+ specification.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document kind, attrs, result carrier, helper, and Tensor method: exact roles and positions,
  formulas, Shapes, types, reduction count, empty/channel-zero cases, saved-output visibility,
  special values, metadata, IDs, failures, and lifecycle boundaries.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [channel glossary](../../../../glossary.md#term-channel); [metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Every [parameter](../../../../glossary.md#term-parameter), non-void result, and expected failure has complete `@param`, `@return`, and
  `@throws` documentation.
- [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) adds the exact method/[carrier](../../../../glossary.md#term-storage-carrier), input/output tables, `[1,2,3]` example, Shape/deferred
  constraints, biased/unbiased distinction, momentum convention, special values, provenance, and
  current-model versus planned compiler/training/runtime boundaries.<span data-glossary-navigation="true"> ([momentum glossary](../../../../glossary.md#term-momentum); [provenance glossary](../../../../glossary.md#provenance); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Compile API records the five-output producer as current model metadata while keeping capture,
  saved-value lifetime, autograd/backward, publication, lowering, and execution planned.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [producer glossary](../../../../glossary.md#term-producer); [capture glossary](../../../../glossary.md#term-capture-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [lowering glossary](../../../../glossary.md#lowering))</span>
- Review glossary terms batch-normalization training, running-statistic transition, batch mean,
  biased/unbiased variance, momentum, saved statistic, and inverse standard deviation; update only
  reusable distinctions.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [statistic glossary](../../../../glossary.md#term-statistic); [mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [saved statistic glossary](../../../../glossary.md#saved-statistic); [standard deviation glossary](../../../../glossary.md#term-standard-deviation))</span>
- Record reasoned no-change conclusions for Runtime and Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- Keep 0021, 0021A, and 0021B Complete; make 0021C Complete only after all acceptance passes;
  keep 0022–0024 Draft without detailed specifications.

## Acceptance criteria

- Exact appended kind, attrs record, fixed 5/5 signature, one receiver, and three-component result
  exist; inference remains fixed 5/1 and public Tensor count is exactly 185.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [inference glossary](../../../../glossary.md#term-inference); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Exact five input and five output roles, public/hidden visibility, saved inverse standard
  deviation, shared producer and indices, and absence of public sibling lookup are locked.<span data-glossary-navigation="true"> ([standard deviation glossary](../../../../glossary.md#term-standard-deviation); [producer glossary](../../../../glossary.md#term-producer); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [Rank](../../../../glossary.md#term-rank)/[axis](../../../../glossary.md#term-axis)/[vector](../../../../glossary.md#term-vector)/static/symbolic [Shape](../../../../glossary.md#shape) rules, all-non-[channel](../../../../glossary.md#term-channel) reduction axes, `C == 0 || N >= 2`
  domain obligation, exact Shape references, and empty behavior match this task.
- Ordered promotion, exact-result-typed momentum/epsilon, computation formats, biased forward and
  unbiased update variance, formulas, special values, reassociation, and determinism pass.<span data-glossary-navigation="true"> ([momentum glossary](../../../../glossary.md#term-momentum); [epsilon glossary](../../../../glossary.md#epsilon); [variance glossary](../../../../glossary.md#term-variance); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Per-output gradient metadata, unresolved layouts, freshness, absent label/storage, exact five-ID
  success, local failure atomicity, and exhaustion behavior pass.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [layouts glossary](../../../../glossary.md#layout); [failure atomicity glossary](../../../../glossary.md#term-failure-atomicity))</span>
- Exact [validation](../../../../glossary.md#term-validation) order and messages pass without [eager](../../../../glossary.md#term-eager-evaluation) value reads or hidden mutation.
- No cross-step/session/checkpoint ownership, compiler/backward/lifetime implementation,
  runtime/publication, algorithm/backend behavior, architecture/dependency/build/cross-module
  change, or later detailed task is introduced.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compiler glossary](../../../../glossary.md#term-compiler); [lifetime glossary](../../../../glossary.md#term-lifetime); [runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- Focused/final model tests, final Javadoc, Markdown link/anchor/fence/newline/whitespace, official
  links, exact 23-path/package/public-surface/status audits, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` pass.
- A separate clean documentation-focused pass finalizes Javadocs/docs/glossary impact and records
  reasoned no-change conclusions before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.normalization.BatchNormTrainingSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormTrainingExpressionTest --tests io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormInferenceExpressionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.datatype.ScalarValueTest --tests io.github.pho001.synaptik.model.shape.ShapeTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Documentation pass after final Javadocs:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass also runs targeted Markdown local-link/anchor/fence/final-newline/trailing-
whitespace checks, verifies the official [ONNX](../../../../glossary.md#term-onnx) and PyTorch links, recalculates the numerical
example, audits exact 23-path [scope](../../../../glossary.md#term-scope)/package placement/public surface/status [synchronization](../../../../glossary.md#term-synchronization), and
records reused Java evidence. Repository-wide validation is deferred to the selected modern-
[operation](../../../../glossary.md#operation) [checkpoint](../../../../glossary.md#term-checkpoint) after task 0022 and [CI](../../../../glossary.md#term-continuous-integration-ci) because this is a [model](../../../../glossary.md#term-model)-only task with no [dependency](../../../../glossary.md#term-dependency),
architecture, shared build, or executable [backend](../../../../glossary.md#backend) change.

## Dependencies

- 0001–0002: [data type](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), static/symbolic, [rank](../../../../glossary.md#term-rank), and [axis](../../../../glossary.md#term-axis) contracts.
- 0005–0007 and 0011–0013: [operation](../../../../glossary.md#operation), [descriptor](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor), factory, and [provenance](../../../../glossary.md#provenance) foundations.
- 0018K: exact attrs-class [signature](../../../../glossary.md#term-signature)/[cardinality](../../../../glossary.md#term-cardinality) [validation](../../../../glossary.md#term-validation).
- 0018L: shared multi-output [producer](../../../../glossary.md#term-producer) and indexed-output construction.
- 0018N: exact typed [scalar](../../../../glossary.md#term-scalar) representation.
- 0018V: correction-zero/one variance, count, computation, empty, special-value, and determinism
  precedent.<span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance))</span>
- 0019B–0019B1: explicit pure state-transition and hidden auxiliary-output precedent.
- 0021–0021B: [normalization](../../../../glossary.md#term-normalization) and stateless [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[inference](../../../../glossary.md#term-inference) contracts.

All dependencies are Complete.

## Follow-up tasks

- Task 0022 remains Draft at its established ID without a detailed specification.
- Task 0023 later owns compiler-generated backward semantics, capture of saved slots three/four,
  and saved-value lifetime. This task does not prescribe the full backward graph.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime); [backward graph glossary](../../../../glossary.md#backward-graph))</span>
- `extensions/training` later owns any session loop, cross-step threading/assignment, parameter or
  buffer ownership, and checkpoint serialization. No detailed extension task is created here.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Runtime/publication and concrete backends later own execution, publication, lowering, algorithms,
  and conformance under their existing plans.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication); [backends glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Task 0024 remains the final [model](../../../../glossary.md#term-model) capability-selection audit.

## Architecture impact

Expected impact: None.

The task composes existing [model](../../../../glossary.md#term-model)-owned typed attrs and shared multi-output [provenance](../../../../glossary.md#provenance) into a pure
semantic state transition. If implementation requires hidden mutation, live cross-step [ownership](../../../../glossary.md#term-ownership-meanings),
a [graph](../../../../glossary.md#graph)-local [identity](../../../../glossary.md#term-identity-meanings), [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend)/extension type, [dependency](../../../../glossary.md#term-dependency) change, or architecture
update, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md.
Implement that task exactly as specified. Do not implement cross-step/session/checkpoint ownership,
compiler backward or saved-value lifetime, runtime/publication, backend algorithms, or later tasks.
Stop and report any architecture, dependency, or maximum-scope conflict. Do not commit or push
unless separately authorized.

After Java implementation and recorded model-test evidence, hand the final diff and evidence to a
separate documentation-focused agent/thread with clean context. That pass must follow the
documentation rules, inspect source/tests, finalize Javadocs, Tensor/Compile APIs, glossary and
planning records, run Javadoc/Markdown/scope/status/whitespace checks, and reuse successful Java
evidence unless executable Java changes.

Update this task's decisions, limitations, evidence, notes, completion summary, and status. Do not
mark Complete before the documentation pass and every acceptance criterion finishes.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Provide the clean documentation context with this task, complete diff, focused/final Java evidence,
five input/output roles, public versus hidden saved slots, formulas and [metadata](../../../../glossary.md#term-metadata), architecture and
23-path constraints, expected [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/glossary/[planning](../../../../glossary.md#term-planning) impact, official [ONNX](../../../../glossary.md#term-onnx)/PyTorch links,
and required [Javadoc](../../../../glossary.md#term-javadoc)/Markdown/[scope](../../../../glossary.md#term-scope)/status evidence. It must inspect source/tests and record
reasoned no-change conclusions rather than relying on the handoff summary.

## Local decisions

- Append `BATCH_NORM_TRAINING` after inference and keep one fixed 5/5 signature. The public method
  is the receiver-only spelling in this task, and <span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [signature glossary](../../../../glossary.md#term-signature))</span>`BatchNormTrainingResult` exposes exactly
  producer slots zero through two.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Preserve slots three and four as real producer-described saved batch mean and inverse standard
  deviation outputs while discarding only their local Tensor wrappers. No sibling lookup or
  public saved-statistic component is added.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [mean glossary](../../../../glossary.md#term-mean); [standard deviation glossary](../../../../glossary.md#term-standard-deviation); [Tensor glossary](../../../../glossary.md#tensor); [lookup glossary](../../../../glossary.md#term-lookup); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- Reduce every non-channel logical axis. Forward normalization uses biased population variance;
  the running-variance transition uses the matching correction-one unbiased estimate. Momentum
  is the new-batch weight, and epsilon occurs only inside the forward inverse-standard-deviation
  square root.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [population variance glossary](../../../../glossary.md#population-variance); [variance glossary](../../../../glossary.md#term-variance); [Momentum glossary](../../../../glossary.md#term-momentum); [weight glossary](../../../../glossary.md#term-weight); [epsilon glossary](../../../../glossary.md#epsilon); [square root glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>
- Treat the exact domain obligation as `C == 0 || N >= 2`. A static zero channel is valid and
  evaluates no output value; a static positive channel rejects reduction count zero or one.
- Keep all five input roles explicit and ordered, promote them in order, and require momentum and
  epsilon to have the exact promoted result type. Construction owns metadata and provenance only.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- The planned 22-path inventory omitted the existing inference semantics test's exact complete
  enum inventory. Appending the mandated kind required its narrow inventory update as the sole
  path-23 exception; its inference signature assertions remain fixed at five inputs and one
  output.

## Known limitations

- The model constructs storage-free expression metadata and does not evaluate normalization,
  select a numerical algorithm, define backend tolerances, or claim executable backend support.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [normalization glossary](../../../../glossary.md#term-normalization); [backend glossary](../../../../glossary.md#backend))</span>
- Compiler capture, proof of unresolved channel/count obligations, saved-output materialization
  and lifetime, autograd/backward construction, publication, liveness, and optimization remain
  planned in their owning compiler tasks.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [channel glossary](../../../../glossary.md#term-channel); [materialization glossary](../../../../glossary.md#materialization); [lifetime glossary](../../../../glossary.md#term-lifetime); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [liveness glossary](../../../../glossary.md#term-liveness))</span>
- Assignment or transport of next statistics across calls, session/step coordination, parameter
  or buffer ownership, and checkpoint serialization remain planned training-extension concerns.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Runtime publication/execution and backend lowering, algorithms, kernels, fusion, and conformance
  remain planned. These accepted boundaries require no follow-up inside task 0021C.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [lowering glossary](../../../../glossary.md#lowering); [kernels glossary](../../../../glossary.md#kernel); [fusion glossary](../../../../glossary.md#term-fusion); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0021c_implementation` ran the exact nine-suite focused command
  listed above after corrections; 50 tests passed with `BUILD SUCCESSFUL in 997ms`. It then ran
  the final `./gradlew :modules:model:test` exactly once; it passed with `BUILD SUCCESSFUL in 1s`.
  Documentation context `/root/task_0021c_implementation/task_0021c_docs` reused this evidence and
  changed no executable Java statements, so it did not repeat either Java test run.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The clean documentation context applied the General, API/Javadoc, Planning, and Example profiles
  and independently reviewed all five production paths, eleven test paths, Tensor/Compile/Runtime/
  Training APIs, glossary, capabilities, task, master plan, roadmap, architecture, and directly
  relevant normalization, scalar, Shape, signature, factory, producer, provenance, reduction,
  explicit-state, and dropout contracts.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [normalization glossary](../../../../glossary.md#term-normalization); [scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape); [signature glossary](../../../../glossary.md#term-signature); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL in 2s`. Generated pages for
  `BatchNormKind`, `BatchNormTrainingAttrs`, `BatchNormTrainingResult`, and [`Tensor`](../../../../glossary.md#tensor) contain the
  finalized kind, scalar, Shape/count, saved-output, and lifecycle contracts. The package-private
  helper correctly has no public generated page.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The targeted Ruby Markdown validator checked the seven documentation/planning paths and resolved
  627 local links, including 167 heading anchors, with zero errors. A separate Ruby check found
  balanced backtick/tilde fences, final newlines, no trailing whitespace, and zero errors in all
  seven paths.
- `curl -I -L --max-time 20` returned HTTP 200 for both official PyTorch BatchNorm1d and ONNX
  BatchNormalization references. Independent Ruby recalculation produced mean <span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx))</span>`2`, squared-
  deviation sum `2`, biased [variance](../../../../glossary.md#term-variance) `2/3`, unbiased variance `1`, inverse [standard deviation](../../../../glossary.md#term-standard-deviation)
  `1.2247356859083902`, next [mean](../../../../glossary.md#term-mean) `8`, next variance `3.25`, and rounded output
  `[-1.949471, 0.500000, 2.949471]`.
- `javap -public` found exactly 185 public `Tensor` methods and exactly one
  `batchNormTraining(int, Tensor, Tensor, Tensor, Tensor, ScalarValue, ScalarValue)` receiver
  returning `BatchNormTrainingResult`. It confirmed the result's exact three record components,
  the attributes' exact three components, and the package-private helper's sole package entry;
  source inspection confirmed the helper is final and field-free with no lookup or overload.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- Final inventory found exactly 23 paths: five production, eleven tests, and seven documentation/
  planning paths. The sole addition beyond the planned 22 is the required existing inference enum-
  inventory lock described above. Production packages remain the planned normalization and tensor
  packages with model-only imports. Status inspection confirmed 0021, 0021A, 0021B, and 0021C
  Complete; 0022–0024 remain Draft; and no detailed 0022–0024 task specification exists.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [model glossary](../../../../glossary.md#term-model))</span>
- Runtime API remains accurate unchanged because this task adds no prepared state, publication,
  storage, scheduling, or execution. Training API remains accurate unchanged because the pure
  returned transition adds no session, parameter/buffer, step, checkpoint, optimizer, or
  assignment ownership. Related normalization/reduction/scalar/Shape/signature/factory/producer/
  provenance contracts remain accurate unchanged because 0021C composes them without modifying
  their APIs or semantics. Architecture/ADRs/tests, backend conformance, integration tests, Gradle,
  other modules, and later tasks require no change because ownership, dependencies, build
  behavior, executable behavior, and cross-module boundaries did not change.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [optimizer glossary](../../../../glossary.md#term-optimizer); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- `git diff --check` passed on the final combined 23-path change.

## Implementation notes

- Added `BATCH_NORM_TRAINING`, [immutable](../../../../glossary.md#term-immutable) `BatchNormTrainingAttrs`, the exact public
  `BatchNormTrainingResult`, one field-free construction helper, and one [canonical](../../../../glossary.md#term-canonical-form) [Tensor](../../../../glossary.md#tensor) receiver.
- The helper validates ordered null/type/rank/axis/vector/channel/count/scalar obligations before
  one five-output factory delegation. It creates exact input/statistic Shapes, promoted output
  types, per-output gradient metadata, shared producer positions, and five ordered identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [vector glossary](../../../../glossary.md#term-vector); [channel glossary](../../../../glossary.md#term-channel); [scalar glossary](../../../../glossary.md#term-scalar); [statistic glossary](../../../../glossary.md#term-statistic); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [producer glossary](../../../../glossary.md#term-producer))</span>
- Focused tests lock kind/[signature](../../../../glossary.md#term-signature)/record/helper/public surface, [Shape](../../../../glossary.md#shape) and deferred constraints,
  `C == 0 || N >= 2`, numerical policies, special values, metadata, provenance, validation order,
  local failure atomicity, freshness, and partial identifier exhaustion.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [failure atomicity glossary](../../../../glossary.md#term-failure-atomicity))</span>
- The documentation pass finalized all affected production Javadocs, Tensor and Compile API
  references, reusable glossary distinctions, capabilities, task evidence, master plan, and
  roadmap while preserving the planned compiler/training/runtime/backend boundaries.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>

## Completion summary

- Completed changes: pure five-input/five-output batch-normalization training semantics, explicit
  next running statistics, hidden producer-described saved values, public expression/result API,
  tests, Javadocs, explanatory documentation, and synchronized planning status.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [producer glossary](../../../../glossary.md#term-producer); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly 23 paths — five production, eleven tests, and seven
  documentation/planning paths; path 23 is the required existing enum-inventory lock.
- Tests and validation: reused passing 50-test focused and final model-suite evidence; final model
  Javadoc, generated-page inspection, 627-link/167-anchor Markdown validation, fences/newlines/
  whitespace, both official links, independent numerical recalculation, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap` public surface,
  exact scope/package/status/later-spec audits, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` all passed.
- Documentation-agent review: completed in clean context
  `/root/task_0021c_implementation/task_0021c_docs` with no executable Java change or duplicate
  Java test run.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Documentation impact: Tensor API, Compile API, glossary, capabilities, this task, model master
  plan, and roadmap finalized. Runtime and Training APIs, related contracts, architecture/ADRs/
  tests, conformance/integration, Gradle, other modules, and later tasks remain accurate unchanged
  for the reasons recorded above.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- Javadoc review: all five production contracts are consistent with source/tests and generated
  successfully; four public pages were inspected and the package-private helper remains internal.
- Glossary impact: added reusable distinctions for batch-normalization training, pure running-
  statistic transition, new-batch-weight momentum, biased-forward/unbiased-running variance, and
  producer-described saved statistics.<span data-glossary-navigation="true"> ([statistic glossary](../../../../glossary.md#term-statistic); [weight glossary](../../../../glossary.md#term-weight); [momentum glossary](../../../../glossary.md#term-momentum); [variance glossary](../../../../glossary.md#term-variance))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0021C; Draft compiler, training-extension, runtime, and backend
  work remains separately owned.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>

Status: Complete
