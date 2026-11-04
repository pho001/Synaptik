# Training statistics and loss expressions

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="batch-normalization-training-and-statistic-transition-expressions"></a>
### Batch-normalization training and statistic-transition expressions

Batch-normalization training uses statistics from the current input occurrence for normalized
output and returns an explicit transition from old running statistics to next running statistics.
It is a pure expression, not a mutable layer or a training/evaluation mode. The sole receiver form
is:

```java
public BatchNormTrainingResult batchNormTraining(
        int channelAxis,
        Tensor scale,
        Tensor bias,
        Tensor runningMean,
        Tensor runningVariance,
        ScalarValue momentum,
        ScalarValue epsilon)
```

The receiver and four Tensor arguments have the same ordered roles and Shape rules as inference:

| Input position | Role | Required Shape |
|---|---|---|
| 0 | receiver input | Rank at least two; exact normalized-output Shape |
| 1 | scale | Exact rank-one `[C]` vector |
| 2 | bias | Exact rank-one `[C]` vector |
| 3 | old running mean | Exact rank-one `[C]` vector |
| 4 | old running variance | Exact rank-one `[C]` vector |

`C` is the extent of the explicitly selected logical channel axis. Every other logical axis,
including every axis before or after it, belongs to the reduction domain; there is no separately
distinguished batch axis or inferred NCHW/NHWC layout. Let `N` be the product of those
non-channel extents. For each channel `c`, the five mathematical outputs are:

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

Forward normalization therefore uses biased population variance, while the running-variance
transition uses the corresponding correction-one unbiased estimate. Momentum is the weight of
the new batch, not the old statistic. Epsilon is added only inside the forward inverse-standard-
deviation square root; it does not alter either variance estimate or the running transition.

The one shared producer describes these exact ordered outputs:

| Output position | Role | Shape | Public wrapper | `requiresGrad` |
|---|---|---|---|---|
| 0 | normalized affine output | Exact input Shape | `result.output()` | `input || scale || bias` |
| 1 | next running mean | Shared rank-one `[C]` Shape | `result.nextRunningMean()` | `input || runningMean` |
| 2 | next running variance | Shared rank-one `[C]` Shape | `result.nextRunningVariance()` | `input || runningVariance` |
| 3 | saved batch mean | Shared rank-one `[C]` Shape | Not exposed | `input` |
| 4 | saved inverse standard deviation | Shared rank-one `[C]` Shape | Not exposed | `input` |

Slots three and four are hidden only at the public wrapper level. Their descriptors and indexed
positions remain reachable through the exact producer retained by each public result. There is no
public sibling-output lookup, and the result record contains exactly the first three outputs.
Current package-private compiler capture creates graph values for all five reachable positions;
saved-value lifetime remains future compiler work. The model does not retain the discarded Tensor
wrappers or define backward formulas.

<a id="numerical-example"></a>
#### Numerical example

The goal is to show the biased-forward/unbiased-running distinction and momentum convention. For
one FLOAT64 channel containing `[1, 2, 3]`, scale `2`, bias `0.5`, old mean `10`, old variance `4`,
momentum `0.25`, and epsilon `1e-5`:

```text
batch mean                  = 2
sum of squared deviations   = 2
biased batch variance       = 2 / 3
unbiased batch variance     = 1
saved inverse std           = 1 / sqrt(2 / 3 + 1e-5)
next running mean           = 0.75 * 10 + 0.25 * 2 = 8
next running variance       = 0.75 * 4 + 0.25 * 1 = 3.25
normalized output           ~= [-1.949471, 0.5, 2.949471]
```

This is a mathematical illustration. Expression construction reads no Tensor values and produces
no numerical result. The convention matches official [PyTorch
BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html). Official
[ONNX BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) also
models explicit running-statistic outputs, but Synaptik does not copy its combined mode, fixed
channel position, scalar defaults, public output surface, or population running-variance update.

<a id="static-checks-deferred-obligations-and-empty-channels"></a>
#### Static checks, deferred obligations, and empty channels

Input rank must be at least two. The positive or negative channel axis is normalized exactly once
against that rank. Scale, bias, old running mean, and old running variance must be exact rank-one
vectors. Structurally equal channel Dimensions pass, unequal static Dimensions fail locally, and
unequal pairs involving an unresolved Dimension defer equality to later compiler binding.

For a non-zero channel extent, the unbiased estimate requires `N >= 2`. The exact static or
deferred obligation is `C == 0 || N >= 2`. A static positive `C` with `N` equal to zero or one
fails locally. A static `C == 0` is valid for any non-negative reduction Shape: all five outputs
are empty and no mean, variance, inverse standard deviation, affine result, or transition is
evaluated. Overflow of a non-zero checked product already proves `N >= 2` and is not stored in
attributes.

All five Tensor inputs must be floating, including FP8. Floating promotion follows
input order. Momentum and epsilon have the exact promoted result type except that FLOAT16/FP8
results use exact FLOAT32 scalar metadata. All four low types and FLOAT32 results use FLOAT32 for
reductions and formula arithmetic; FLOAT64 results use FLOAT64. Final low outputs are narrowed
once to the result format. The operation selects no traversal, pass count, reassociation,
compensation method, kernel, or backend algorithm.

<a id="special-values-identity-effects-and-lifecycle-boundary"></a>
#### Special values, identity effects, and lifecycle boundary

Construction reads no values and does not require old running variance to be non-negative. A NaN
or infinity in a non-empty batch domain makes that channel's batch mean, both variance estimates,
saved inverse standard deviation, and normalized outputs NaN. A finite constant domain has
positive-zero variances and a positive saved inverse standard deviation. Scale, bias, old
statistics, signed zero, overflow, and underflow otherwise follow ordinary floating arithmetic.
The written momentum formula applies even at zero and one, so a zero coefficient does not suppress
NaN or infinity. A conforming implementation must not use unstable cancellation to invent a
negative finite variance, but may otherwise reassociate within future tolerance. Bitwise equality,
NaN payload/sign preservation, and identical cross-backend rounding are not promised.

Every success creates five fresh, unlabeled, storage-free, unresolved-layout Tensor outputs and
five IDs in output order under one identity-distinct producer. Local validation failures occur
before factory delegation and consume no ID. Identifier exhaustion may leave IDs for earlier
output positions consumed, returns no partial result, and does not mutate a running statistic.

These are current model metadata facts only. Package-private compiler capture and proof or
retention of deferred constraints are current. Saved-value materialization and lifetime,
autograd/backward construction, publication, liveness, graph optimization, concrete binding, and
public compilation remain planned compiler or lifecycle work. Cross-step statistic assignment,
session and checkpoint ownership remain planned training-extension work. Runtime publication and
execution remain planned runtime work, while lowering, algorithms, kernels, and tolerances remain
backend work. No layer, buffer owner, training mode, in-place update, or state across calls is
created.

<a id="mean-squared-error-loss-expressions"></a>
### Mean-squared-error loss expressions

Mean-squared error (MSE) compares a prediction Tensor with a target Tensor at the same logical
coordinates. The sole current receiver form requires an explicit loss reduction:

```java
public Tensor meanSquaredError(Tensor target, LossReduction reduction)
```

The receiver is the prediction at producer input position zero, and `target` is input position
one. Both inputs must be floating, including FP8. Their data types are promoted in
that order with floating promotion; no cast Tensor is inserted. Prediction and target must have equal
rank and positionally compatible Dimensions. Structurally equal Dimensions pass, unequal static
Dimensions fail, and an unequal pair involving an unresolved Dimension defers an equality
obligation to later compiler capture or concrete binding. No scalar, singleton, leading-axis,
right-aligned, or other target broadcasting is accepted.

For each logical coordinate `i`, the selected mathematical loss is:

```text
d_i    = prediction_i - target_i
loss_i = d_i * d_i
```

Let `E` be the complete logical element count after every unresolved Dimension is bound. The
explicit reduction determines both output and Shape:

| Reduction | Result | Result Shape |
|---|---|---|
| `LossReduction.NONE` | One `loss_i` for every logical coordinate | The exact prediction Shape reference |
| `LossReduction.SUM` | `sum_i(loss_i)` | Shared `Shape.scalar()` |
| `LossReduction.MEAN` | `sum_i(loss_i) / E` | Shared `Shape.scalar()` |

`E` counts every logical coordinate, not only a batch or leading axis. A scalar has `E = 1`. A
Shape with any zero extent has `E = 0`: `NONE` is empty, `SUM` is positive zero, and `MEAN` is
NaN because it is the selected zero-over-zero empty mean.

For prediction `[1, 2, 4]` and target `[1, 4, 1]`, the coordinate losses are `[0, 4, 9]`, `SUM`
is `13`, and `MEAN` is `13 / 3`. This mathematical example explains the metadata request;
constructing the expression reads no values. It is comparable to official
[PyTorch MSE loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html),
while Synaptik deliberately has no default reduction, weights, or broadcasting.

All four low types and FLOAT32 results use FLOAT32 subtraction, multiplication, accumulation, and
division; FLOAT64 results use FLOAT64. Final values are rounded to the promoted result type.
Implementations may use equal-or-wider intermediates, stable or compensated sums, vectorization, parallelization,
fusion, and reassociation when later tolerances and the required special-value classes remain
intact. No fixed traversal, bitwise cross-backend identity, finite rounding identity, or NaN
payload/sign preservation is promised.

NaN in either operand produces NaN squared error. Equal-sign infinities subtract to NaN;
opposite-sign infinities or one finite and one infinite operand square to positive infinity.
Every exact zero difference, including any signed-zero pairing, squares to positive zero. Finite
subtraction or squaring may overflow to positive infinity or underflow to positive zero. A
reduced NaN propagates; otherwise any positive infinity makes the reduced result positive
infinity, and an all-finite exact-zero domain reduces to positive zero before mean division. An
implementation must not replace `(p - t) * (p - t)` with an algebraic expansion that changes
these value classes.

Every success creates one fresh, unlabeled, storage-free, unresolved-layout Tensor with one fresh
producer, one Tensor ID, and provenance output index zero. The result's `requiresGrad` metadata is
the logical OR of prediction and target eligibility. Repeated equal calls remain identity-
distinct. No input is mutated, and local null, type, rank, or static-Dimension failures occur
before factory delegation and consume no result ID.

These are current model metadata facts and do not define a Model-owned gradient rule.
Package-private compiler first-order autograd now supports prediction and target roles for all
three reductions, restores reduced cotangents through logical Tensor element counts, and uses
exact typed scalar-operation coefficients `2` and `-2`. Compiler capture and deferred equality
proof are current. CPU MSE includes all four low types with FLOAT32 working arithmetic and
final logical narrowing. Metal owns the admitted homogeneous BFLOAT16/FLOAT16/FLOAT32
canonical positive-rank forward domain for `NONE`, `SUM`, and `MEAN`, with the exact input-gradient OR.
Generated backward ownership requires each actual derived occurrence to qualify independently.
Every other Metal loss or normalization family remains fail-closed.
Broader decomposition, backend support, runtime behavior, and training-session coordination remain separate
responsibilities.

<a id="categorical-cross-entropy-with-logits-expressions"></a>
### Categorical-cross-entropy-with-logits expressions

Categorical cross entropy compares each logits slice with either a full floating target slice or
one integral class index. A *logit* is an unnormalized class score. The existing three-argument
receiver dispatches by the target's exact data type, and one four-argument form adds an exact typed
ignore index for integral targets:

```java
public Tensor categoricalCrossEntropyWithLogits(
        Tensor target, int classAxis, LossReduction reduction)

public Tensor categoricalCrossEntropyWithLogits(
        Tensor target,
        int classAxis,
        LossReduction reduction,
        ScalarValue ignoreIndex)
```

| Form and target type | Meaning | Ignore value |
|---|---|---|
| Three arguments, floating types including FP8 | Existing dense target-weighted loss | None |
| Three arguments, INT32/INT64 | Index-target selected-class loss | None |
| Four arguments, INT32/INT64 | Index-target selected-class loss | Required `ScalarValue` with the exact target type |

BOOL is unsupported. The four-argument form does not accept a floating target, and no public form
accepts `Optional`, a nullable ignore value, or an implicit cast. Both meanings record ordered
producer inputs `[logits, target]`; the ignore value is immutable operation attributes metadata,
not a third Tensor input.

The dense branch is unchanged: logits and target have equal rank and positionally compatible
Dimensions, both are floating, floating promotion selects the result type and combined gradient
eligibility, and broadcasting is forbidden. For group `g` outside the normalized class axis:

```text
m[g]    = max_c(logits[g,c])
lse[g]  = m[g] + log(sum_c(exp(logits[g,c] - m[g])))
loss[g] = sum_c(target[g,c] * (lse[g] - logits[g,c]))
```

Dense target slices remain caller-obligated to be finite, non-negative, and normalized to one.
Construction reads no values and neither diagnoses nor renormalizes them. An exact zero target
weight contributes positive zero even when its log probability is negative infinity.
BFLOAT16/FLOAT16/FLOAT32 dense results use FLOAT32 computation after floating promotion and
FLOAT64 results use FLOAT64. A NaN or positive infinity in a participating logits slice and an
all-negative-infinity slice produce NaN. With at least one finite logit, positive target weight on
negative infinity contributes positive infinity, while exact zero weight still contributes
positive zero.

An index target has exact type INT32 or INT64 and Shape equal to the logits Shape with the class
axis removed. Target axis `t` maps to logits axis `t` before the class axis and `t + 1` after it.
Unequal static mapped Dimensions fail; an unequal pair involving an unresolved Dimension leaves a
later equality obligation. No broadcasting, target promotion, one-hot conversion, negative-index
wrapping, or clamping occurs. `NONE` retains the exact target Shape object, including scalar Shape
for rank-one logits. Index results retain exact logits type and use only logits
`requiresGrad` metadata.

For index target `y[g]` and optional ignore value `I`, the requested mathematical meaning is:

```text
if I is present and y[g] == I:
    loss[g] = +0
else:
    require 0 <= y[g] < classExtent
    loss[g] = lse[g] - logits[g,y[g]]
```

Exact ignore comparison occurs before bounds checking or logits evaluation, so the ignore value
may itself be out of range. Every non-matching target must eventually be in range. Model
construction reads no target or logits values—even from eager tensors—so bounds proof and safe
indexed access remain later compiler, preparation, or execution obligations.

For logits `[1, 2, 3]`, `lse` is approximately `3.407605964`: index target `2` gives
`0.407605964`, index target `0` gives `2.407605964`, and INT32 target `-1` with exact
`ScalarValue.int32(-1)` gives positive zero without evaluating the logits slice. The same logits
with dense target `[0, 0, 1]` give `0.407605964`. These are mathematical examples, not values
computed while constructing a Tensor expression. The meanings are comparable to official
[PyTorch cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
and [ONNX SoftmaxCrossEntropyLoss](https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html).

Let `S` be the target/sample-domain size and `N` the non-ignored target count (`N = S` when no
ignore value is present):

| Reduction | Dense result | Index result | Result Shape |
|---|---|---|---|
| `LossReduction.NONE` | One weighted loss per group | One selected loss per target; ignored positions are `+0` | Class-axis-removed Shape; the index branch retains the exact target Shape |
| `LossReduction.SUM` | Sum over all groups | Sum over non-ignored targets | Shared `Shape.scalar()` |
| `LossReduction.MEAN` | Sum divided by `S` | Sum divided by `N` | Shared `Shape.scalar()` |

An empty domain produces empty `NONE`, positive-zero `SUM`, and NaN `MEAN`. An all-ignored,
non-empty index domain produces positive zeros for `NONE`, positive-zero `SUM`, and NaN `MEAN`.
Without ignore, a non-empty domain requires positive class extent. With ignore, zero class extent
is also valid when every target is ignored; because construction reads no values, it retains the
deferred alternative `S == 0 || classExtent > 0 || all targets equal ignoreIndex`. Every
non-ignored target is invalid when class extent is zero.

All four low types and FLOAT32 index computation uses FLOAT32 max, exponential, summation,
logarithm, subtraction, loss accumulation, and division; FLOAT64 uses FLOAT64. Final values round to logits
type. Equal-or-wider intermediates, compensation, vectorization, fusion, and reassociation remain
allowed. No narrower arithmetic, fixed traversal, bitwise identity, NaN payload/sign, or identical
finite rounding across backends is promised. For a non-ignored slice, any NaN or positive infinity
and an all-negative-infinity slice produce NaN. With at least one finite logit, selecting negative
infinity produces positive infinity; selecting a finite class gives its finite stable loss.
Ignored positions stay positive zero without propagating special values. Participating NaN
propagates through reductions; otherwise participating positive infinity produces positive
infinity.

Every successful request creates one fresh unlabeled, storage-free Tensor with unresolved layout,
one producer, one ID, and output-index-zero provenance over `[logits, target]`. It creates no
softmax, gather, select, cast, arithmetic, reduction, or one-hot sub-producer and mutates no input.
These are current model-expression facts only. Package-private compiler capture and
descriptor-visible deferred proof are current. Compiler-owned first-order autograd supports
logits and target roles for dense floating targets. For index targets it supports logits only,
requires positive static class depth, clamps ignored targets before one-hot construction, and
excludes ignored rows through a final selection; dynamic or zero class depth fails closed.
Constant analysis, value-dependent bounds and target-content enforcement, concrete binding,
decomposition, optimization, backend lowering, preparation, numerical execution, runtime
publication, and training coordination remain planned in their owning lifecycle layers.


---

[Previous topic](general-scan.md) · [Next topic](reshape-expand.md)
