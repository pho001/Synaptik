# Training statistics and [loss](../../glossary.md#loss) expressions

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## State transitions and objectives are explicit expression inputs

Application or NN layer-building code calls the Tensor methods with prediction/input Tensors,
target/statistic Tensors and fixed typed attributes. Construction retains exact Tensor references
and returns descriptions; it does not copy numeric data, update a running statistic or perform a
training step. For batch normalization, the application or layer must explicitly consume the next
running-statistic results. For a loss, Engine/Training later evaluates the selected objective.
Use [normalization inference](normalization.md#batch-normalization-inference-expressions) when
statistics are already known, and [reduction domains](reductions.md) to interpret SUM versus MEAN.

**[Reader model](../../glossary.md#term-reader-model).** A **[statistic](../../glossary.md#term-statistic)** summarizes a set of values. [Batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [mean](../../glossary.md#term-mean) and [variance](../../glossary.md#term-variance) summarize
the current input; **running statistics** summarize prior inputs and have explicit next values
here, rather than being silently mutated. Variance averages squared deviations from the mean;
the biased population form divides by the count, while the correction-one estimate divides by
one less than that count. **[Momentum](../../glossary.md#term-momentum)** below is the [weight](../../glossary.md#term-weight) of the new batch in that update,
not an [optimizer](../../glossary.md#term-optimizer)'s momentum [parameter](../../glossary.md#term-parameter).

A **loss** measures disagreement with a [target](../../glossary.md#term-target). Mean squared error measures squared numeric
differences; [categorical cross entropy](../../glossary.md#categorical-cross-entropy) measures disagreement over alternative classes.
**Logits** are unrestricted class scores before probability [normalization](../../glossary.md#term-normalization), not probabilities
already summing to one. A [dense target](../../glossary.md#dense-target) supplies a weight for each class; an [index target](../../glossary.md#index-target) names
one class. The exact count, ignored-target, empty-domain, and supported-[gradient](../../glossary.md#term-gradient) rules below
remain specific to each loss.

<a id="batch-normalization-training-and-statistic-transition-expressions"></a>
### [Batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../glossary.md#term-normalization) training and [statistic](../../glossary.md#term-statistic)-transition expressions

Batch-normalization training uses statistics from the current input [occurrence](../../glossary.md#term-occurrence) for normalized
output and returns an explicit transition from old running statistics to next running statistics.
It is a pure expression, not a [mutable](../../glossary.md#term-mutable) layer or a training/[evaluation](../../glossary.md#term-evaluation) mode. The sole receiver form
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

The receiver and four [Tensor](../../glossary.md#tensor) [arguments](../../glossary.md#term-argument) have the same ordered roles and [Shape](../../glossary.md#shape) rules as [inference](../../glossary.md#term-inference):

| Input position | Role | Required Shape |
|---|---|---|
| 0 | receiver input | [Rank](../../glossary.md#term-rank) at least two; exact normalized-output Shape |
| 1 | scale | Exact [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) `[C]` [vector](../../glossary.md#term-vector) |
| 2 | [bias](../../glossary.md#term-bias) | Exact rank-one `[C]` vector |
| 3 | old running [mean](../../glossary.md#term-mean) | Exact rank-one `[C]` vector |
| 4 | old running [variance](../../glossary.md#term-variance) | Exact rank-one `[C]` vector |

`C` is the [extent](../../glossary.md#term-extent) of the explicitly selected logical [channel](../../glossary.md#term-channel) [axis](../../glossary.md#term-axis). Every other logical axis,
including every axis before or after it, belongs to the reduction domain; there is no separately
distinguished batch axis or inferred [NCHW](../../glossary.md#term-nchw)/[NHWC](../../glossary.md#term-nhwc) [layout](../../glossary.md#layout). Let `N` be the product of those
non-channel extents. For each channel `c`, the five mathematical outputs are:

For a channel $0\leq c<C$, enumerate its $N\geq2$ contributors as finite real
$x_{c,i}$, $0\leq i<N$. Define batch mean $\mu_c$, squared-deviation sum $S_c$,
population variance $v_c$, correction-one estimate $\hat v_c$, and saved inverse deviation $r_c$:

$$
\mu_c=\frac1N\sum_{i=0}^{N-1}x_{c,i},\quad
S_c=\sum_{i=0}^{N-1}(x_{c,i}-\mu_c)^2,\quad
v_c=S_c/N,\quad \hat v_c=S_c/(N-1),\quad
r_c=1/\sqrt{v_c+\varepsilon}.
$$

Let $\gamma_c,\beta_c$ be scale and bias, $m_c,w_c$ old running mean and variance,
$\lambda$ the new-batch momentum, and $\varepsilon>0$ epsilon. The output and next statistics are

$$
y_{c,i}=((x_{c,i}-\mu_c)r_c)\gamma_c+\beta_c,\qquad
m'_c=(1-\lambda)m_c+\lambda\mu_c,\qquad
w'_c=(1-\lambda)w_c+\lambda\hat v_c.
$$

Variance, $S_c$ and epsilon have squared input units. These finite-real formulas explain the
roles; actual arithmetic retains the written momentum terms even at endpoints and the saved
rounding boundaries. The zero-channel case below evaluates none of these formulas.

Forward normalization therefore uses biased [population variance](../../glossary.md#population-variance), while the running-variance
transition uses the corresponding correction-one unbiased estimate. [Momentum](../../glossary.md#term-momentum) is the [weight](../../glossary.md#term-weight) of
the new batch, not the old statistic. [Epsilon](../../glossary.md#epsilon) is added only inside the forward inverse-standard-
deviation [square root](../../glossary.md#norms-elementary-functions-and-probability-distributions); it does not alter either variance estimate or the running transition.

The one shared [producer](../../glossary.md#term-producer) describes these exact ordered outputs:

| Output position | Role | Shape | Public wrapper | `requiresGrad` |
|---|---|---|---|---|
| 0 | normalized affine output | Exact input Shape | `result.output()` | `input || scale || bias` |
| 1 | next running mean | Shared rank-one `[C]` Shape | `result.nextRunningMean()` | `input || runningMean` |
| 2 | next running variance | Shared rank-one `[C]` Shape | `result.nextRunningVariance()` | `input || runningVariance` |
| 3 | saved batch mean | Shared rank-one `[C]` Shape | Not exposed | `input` |
| 4 | saved inverse [standard deviation](../../glossary.md#term-standard-deviation) | Shared rank-one `[C]` Shape | Not exposed | `input` |

Slots three and four are omitted from the ergonomic result record, not discarded. The producer
retains their [canonical](../../glossary.md#term-canonical-form) Tensor wrappers, reachable through `producer.output(3)` and
`producer.output(4)`. [Compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings) creates all five [graph](../../glossary.md#graph) positions, and normalization [AD](../../glossary.md#autograd)
consumes the same-occurrence saved mean and inverse standard deviation rather than recomputing
statistics. Their physical [liveness](../../glossary.md#term-liveness) and storage belong to the compiled/[prepared execution](../../glossary.md#prepared-execution--preparedexecution),
not to [Model](../../glossary.md#term-model)-owned [host storage](../../glossary.md#host-storage).

<a id="numerical-example"></a>
#### Numerical example

The goal is to show the biased-forward/unbiased-running distinction and [momentum](../../glossary.md#term-momentum) convention. For
one [FLOAT64](../../glossary.md#term-float64) [channel](../../glossary.md#term-channel) containing `[1, 2, 3]`, scale `2`, [bias](../../glossary.md#term-bias) `0.5`, old [mean](../../glossary.md#term-mean) `10`, old [variance](../../glossary.md#term-variance) `4`,
momentum `0.25`, and [epsilon](../../glossary.md#epsilon) `1e-5`:

Here $\mu=2$, $S=2$, $v=2/3$, $\hat v=1$ and
$r=1/\sqrt{2/3+10^{-5}}$. The transitions are

$$
m'=0.75(10)+0.25(2)=8,\qquad w'=0.75(4)+0.25(1)=3.25.
$$

The normalized affine output is approximately `[-1.949471, 0.5, 2.949471]`.
The different variance denominators explain why the running variance moves toward $1$, not $2/3$.

This is a mathematical illustration. Expression construction reads no [Tensor](../../glossary.md#tensor) values and produces
no numerical result. The convention matches official [PyTorch
BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html). Official
[ONNX BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html)<span data-glossary-navigation="true"> ([ONNX glossary](../../glossary.md#term-onnx))</span> also
models explicit running-[statistic](../../glossary.md#term-statistic) outputs, but Synaptik does not copy its combined mode, fixed
channel position, [scalar](../../glossary.md#term-scalar) defaults, public output surface, or population running-variance update.

<a id="static-checks-deferred-obligations-and-empty-channels"></a>
#### Static checks, deferred obligations, and empty channels

Input [rank](../../glossary.md#term-rank) must be at least two. The positive or negative [channel](../../glossary.md#term-channel) [axis](../../glossary.md#term-axis) is normalized exactly once
against that rank. Scale, [bias](../../glossary.md#term-bias), old running [mean](../../glossary.md#term-mean), and old running [variance](../../glossary.md#term-variance) must be exact [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix)
[vectors](../../glossary.md#term-vector). Structurally equal channel [Dimensions](../../glossary.md#dimension) pass, unequal static Dimensions fail locally, and
unequal pairs involving an unresolved Dimension defer equality to later [compiler](../../glossary.md#term-compiler) [binding](../../glossary.md#term-binding).

For a non-zero channel [extent](../../glossary.md#term-extent), the unbiased estimate requires `N >= 2`. The exact static or
[deferred obligation](../../glossary.md#term-deferred-obligation) is `C == 0 || N >= 2`. A static positive `C` with `N` equal to zero or one
fails locally. A static `C == 0` is valid for any non-negative reduction [Shape](../../glossary.md#shape): all five outputs
are empty and no mean, variance, inverse [standard deviation](../../glossary.md#term-standard-deviation), affine result, or transition is
evaluated. [Overflow](../../glossary.md#term-overflow) of a non-zero checked product already proves `N >= 2` and is not stored in
attributes.

All five [Tensor](../../glossary.md#tensor) inputs must be floating, including [FP8](../../glossary.md#term-fp8). Floating promotion follows
input order. [Momentum](../../glossary.md#term-momentum) and [epsilon](../../glossary.md#epsilon) have the exact promoted result type except that [FLOAT16](../../glossary.md#term-float16)/FP8
results use exact [FLOAT32](../../glossary.md#term-float32) [scalar](../../glossary.md#term-scalar) [metadata](../../glossary.md#term-metadata). All four low types and FLOAT32 results use FLOAT32 for
reductions and formula arithmetic; [FLOAT64](../../glossary.md#term-float64) results use FLOAT64. Final low outputs are narrowed
once to the result format. The [operation](../../glossary.md#operation) selects no traversal, pass count, [reassociation](../../glossary.md#term-reassociation),
compensation method, [kernel](../../glossary.md#kernel), or [backend](../../glossary.md#backend) algorithm.

<a id="special-values-identity-effects-and-lifecycle-boundary"></a>
#### Special values, [identity](../../glossary.md#term-identity-meanings) effects, and [lifecycle](../../glossary.md#lifecycle) boundary

Construction reads no values and does not require old running [variance](../../glossary.md#term-variance) to be non-negative. A [NaN](../../glossary.md#term-nan)
or [infinity](../../glossary.md#term-infinity) in a non-empty [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) domain makes that [channel](../../glossary.md#term-channel)'s batch [mean](../../glossary.md#term-mean), both variance estimates,
saved inverse [standard deviation](../../glossary.md#term-standard-deviation), and normalized outputs NaN. A [finite](../../glossary.md#term-finite) [constant](../../glossary.md#term-constant) domain has
positive-zero variances and a positive saved inverse standard deviation. Scale, [bias](../../glossary.md#term-bias), old
statistics, [signed zero](../../glossary.md#term-signed-zero), [overflow](../../glossary.md#term-overflow), and [underflow](../../glossary.md#term-underflow) otherwise follow ordinary floating arithmetic.
The written [momentum](../../glossary.md#term-momentum) formula applies even at zero and one, so a zero coefficient does not suppress
NaN or infinity. A conforming implementation must not use unstable [cancellation](../../glossary.md#term-cancellation) to invent a
negative finite variance, but may otherwise reassociate within future [tolerance](../../glossary.md#term-tolerance). Bitwise equality,
[NaN payload](../../glossary.md#nan-infinity-and-signed-zero)/sign preservation, and identical cross-[backend](../../glossary.md#backend) [rounding](../../glossary.md#term-rounding) are not promised.

Every success creates five fresh, unlabeled, storage-free, unresolved-[layout](../../glossary.md#layout) [Tensor](../../glossary.md#tensor) outputs and
five IDs in output order under one identity-distinct [producer](../../glossary.md#term-producer). Local [validation](../../glossary.md#term-validation) failures occur
before factory delegation and consume no ID. Identifier exhaustion may leave IDs for earlier
output positions consumed, returns no partial result, and does not mutate a [running statistic](../../glossary.md#term-running-statistic).

These are [metadata](../../glossary.md#term-metadata)-construction effects, not a denial of downstream support. [Compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings),
deferred proof, selected first-order [derivatives](../../glossary.md#term-derivative) and saved-output use are implemented.
For output zero, [normalization](../../glossary.md#term-normalization) [AD](../../glossary.md#autograd) selects input/scale/bias; outputs one and two select
input/running-mean and input/running-variance respectively. Qualified backend [owners](../../glossary.md#term-owner) provide
[lowering](../../glossary.md#lowering), preparation, saved-value storage, execution and [publication](../../glossary.md#term-publication). The constructor itself
creates no layer, mode, [buffer](../../glossary.md#term-buffer) [owner](../../glossary.md#term-owner), in-place [statistic](../../glossary.md#term-statistic) update or cross-call state.
Callers must explicitly consume the next-statistic results; constructing a transition does
not automatically assign it to an [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) buffer or [checkpoint](../../glossary.md#term-checkpoint) an [optimizer](../../glossary.md#term-optimizer).

<a id="mean-squared-error-loss-expressions"></a>
### [Mean](../../glossary.md#term-mean)-squared-error [loss](../../glossary.md#loss) expressions

[Mean-squared error](../../glossary.md#mean-squared-error--mse) ([MSE](../../glossary.md#mean-squared-error--mse)) compares a [prediction](../../glossary.md#term-prediction) [Tensor](../../glossary.md#tensor) with a [target](../../glossary.md#term-target) Tensor at the same logical
[coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix). The sole current receiver form requires an explicit [loss reduction](../../glossary.md#loss-reduction):

```java
public Tensor meanSquaredError(Tensor target, LossReduction reduction)
```

The receiver is the prediction at [producer](../../glossary.md#term-producer) input position zero, and `target` is input position
one. Both inputs must be floating, including [FP8](../../glossary.md#term-fp8). Their data types are promoted in
that order with floating promotion; no cast Tensor is inserted. Prediction and target must have equal
[rank](../../glossary.md#term-rank) and positionally compatible [Dimensions](../../glossary.md#dimension). Structurally equal Dimensions pass, unequal static
Dimensions fail, and an unequal pair involving an unresolved Dimension defers an equality
obligation to later [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings) or concrete [binding](../../glossary.md#term-binding). No [scalar](../../glossary.md#term-scalar), [singleton](../../glossary.md#term-singleton), leading-[axis](../../glossary.md#term-axis),
right-aligned, or other target [broadcasting](../../glossary.md#broadcasting) is accepted.

For a finite-real prediction $p_i$ and target $t_i$ at matching coordinate $i$, define
difference $d_i$ and coordinate loss $\ell_i$:

$$
d_i=p_i-t_i,\qquad \ell_i=d_i^2.
$$

Here $i$ enumerates every logical position, not only batch rows. The squared loss has squared
input units. Actual evaluation retains subtraction followed by multiplication; the real
square notation does not authorize expansion into separate squares and cross terms.

Let `E` be the complete logical [element](../../glossary.md#term-element) count after every unresolved Dimension is bound. The
explicit reduction determines both output and [Shape](../../glossary.md#shape):

| Reduction | Result | Result Shape |
|---|---|---|
| `LossReduction.NONE` | One `loss_i` for every logical coordinate | The exact prediction Shape reference |
| `LossReduction.SUM` | $\sum_{i=0}^{E-1}\ell_i$ | Shared `Shape.scalar()` |
| `LossReduction.MEAN` | $\frac1E\sum_{i=0}^{E-1}\ell_i$ for $E>0$ | Shared `Shape.scalar()` |

`E` counts every logical coordinate, not only a [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) or leading axis. A scalar has `E = 1`. A
Shape with any zero [extent](../../glossary.md#term-extent) has `E = 0`: `NONE` is empty, `SUM` is [positive zero](../../glossary.md#nan-infinity-and-signed-zero), and `MEAN` is
[NaN](../../glossary.md#term-nan) because it is the selected zero-over-zero empty mean.

For prediction `[1, 2, 4]` and target `[1, 4, 1]`, coordinate losses are `[0, 4, 9]`;
SUM is $0+4+9=13$ and MEAN is $13/3$. The denominator counts all three positions.
This mathematical example explains the request; constructing the expression reads no values.
It is comparable to official
[PyTorch MSE loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html),
while Synaptik deliberately has no default reduction, [weights](../../glossary.md#term-weight), or broadcasting.

All four low types and [FLOAT32](../../glossary.md#term-float32) results use FLOAT32 subtraction, multiplication, accumulation, and
division; [FLOAT64](../../glossary.md#term-float64) results use FLOAT64. Final values are rounded to the promoted result type.
The [primitive and aggregate floors](../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives)
govern each arithmetic site, contributor, mandatory divisor and observable boundary.
There is no blanket equal-or-wider-intermediate or arbitrary-[fusion](../../glossary.md#term-fusion) permission. [Finite](../../glossary.md#term-finite)
tolerances qualify routes in tests; they neither replace these rules nor define a production
whole-output [error envelope](../../glossary.md#term-error-envelope). Cross-[backend](../../glossary.md#backend) bitwise [identity](../../glossary.md#term-identity-meanings) and [NaN payload](../../glossary.md#nan-infinity-and-signed-zero)/sign are not promised.

NaN in either operand produces NaN squared error. Equal-sign [infinities](../../glossary.md#nan-infinity-and-signed-zero) subtract to NaN;
opposite-sign infinities or one finite and one infinite operand square to [positive infinity](../../glossary.md#nan-infinity-and-signed-zero).
Every exact zero difference, including any [signed-zero](../../glossary.md#nan-infinity-and-signed-zero) pairing, squares to positive zero. Finite
subtraction or squaring may [overflow](../../glossary.md#term-overflow) to positive infinity or [underflow](../../glossary.md#term-underflow) to positive zero. A
reduced NaN propagates; otherwise any positive infinity makes the reduced result positive
infinity, and an all-finite exact-zero domain reduces to positive zero before mean division. An
implementation must not replace `(p - t) * (p - t)` with an algebraic expansion that changes
these value classes.

Every success creates one fresh, unlabeled, storage-free, unresolved-[layout](../../glossary.md#layout) Tensor with one fresh
producer, one Tensor ID, and [provenance](../../glossary.md#provenance) output index zero. The result's `requiresGrad` metadata is
the logical OR of prediction and target eligibility. Repeated equal calls remain identity-
distinct. No input is mutated, and local null, type, rank, or static-Dimension failures occur
before factory delegation and consume no result ID.

These are current [model](../../glossary.md#term-model) metadata facts and do not define a Model-owned [gradient](../../glossary.md#term-gradient) rule.
Package-private compiler first-order [autograd](../../glossary.md#autograd) now supports prediction and target roles for all
three reductions, restores reduced cotangents through logical Tensor element counts, and uses
exact typed scalar-[operation](../../glossary.md#operation) coefficients `2` and `-2`. Compiler capture and deferred equality
proof are current. [CPU](../../glossary.md#term-cpu) MSE includes all four low types with FLOAT32 [working arithmetic](../../glossary.md#accumulator-type) and
final logical [narrowing](../../glossary.md#term-narrowing). [Metal](../../glossary.md#term-metal) owns the admitted [homogeneous](../../glossary.md#term-homogeneous) [BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16)/FLOAT32
[canonical](../../glossary.md#term-canonical-form) positive-rank forward domain for `NONE`, `SUM`, and `MEAN`, with the exact input-gradient OR.
Generated backward [ownership](../../glossary.md#term-ownership-meanings) requires each actual derived [occurrence](../../glossary.md#term-occurrence) to qualify independently.
Every other Metal loss or [normalization](../../glossary.md#term-normalization) family remains [fail-closed](../../glossary.md#term-fail-closed).
Broader [decomposition](../../glossary.md#term-decomposition), backend support, [runtime](../../glossary.md#term-runtime) behavior, and training-session coordination remain separate
responsibilities.

<a id="categorical-cross-entropy-with-logits-expressions"></a>
### Categorical-cross-entropy-with-logits expressions

[Categorical cross entropy](../../glossary.md#categorical-cross-entropy) compares each logits [slice](../../glossary.md#slice) with either a full floating [target](../../glossary.md#term-target) slice or
one integral class index. A *[logit](../../glossary.md#logit)* is an unnormalized class score. The existing three-[argument](../../glossary.md#term-argument)
receiver dispatches by the target's exact [data type](../../glossary.md#data-type--datatype), and one four-argument form adds an exact typed
[ignore index](../../glossary.md#ignore-index) for integral targets:

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
| Three arguments, floating types including [FP8](../../glossary.md#term-fp8) | Existing [dense target](../../glossary.md#dense-target)-weighted [loss](../../glossary.md#loss) | None |
| Three arguments, [INT32](../../glossary.md#data-type--datatype)/[INT64](../../glossary.md#data-type--datatype) | Index-target selected-class loss | None |
| Four arguments, INT32/INT64 | Index-target selected-class loss | Required [`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue) with the exact target type |

[BOOL](../../glossary.md#data-type--datatype) is unsupported. The four-argument form does not accept a floating target, and no public form
accepts `Optional`, a nullable ignore value, or an implicit cast. Both meanings record ordered
[producer](../../glossary.md#term-producer) inputs `[logits, target]`; the ignore value is [immutable](../../glossary.md#term-immutable) [operation](../../glossary.md#operation) attributes [metadata](../../glossary.md#term-metadata),
not a third [Tensor](../../glossary.md#tensor) input.

The dense branch is unchanged: logits and target have equal [rank](../../glossary.md#term-rank) and positionally compatible
[Dimensions](../../glossary.md#dimension), both are floating, floating promotion selects the result type and combined [gradient](../../glossary.md#term-gradient)
eligibility, and [broadcasting](../../glossary.md#broadcasting) is forbidden. For group `g` outside the normalized [class axis](../../glossary.md#class-axis):

Let $C>0$ be class extent; $c$ ranges over integer classes $0\leq c<C$, while $g$
denotes all coordinates outside that axis. For finite real logits $z_{g,c}$ and valid dense
target weights $t_{g,c}$, define maximum $m_g$, log-sum-exp $a_g$ and group loss $\ell_g$:

$$
m_g=\max_{0\leq c<C}z_{g,c},\qquad
a_g=m_g+\log\left(\sum_{c=0}^{C-1}\exp(z_{g,c}-m_g)\right),\qquad
\ell_g=\sum_{c=0}^{C-1}t_{g,c}(a_g-z_{g,c}).
$$

The maximum shift expresses the intended class calculation; it is not a license to replace
literal arithmetic sites elsewhere (such as attention) or ignore the exceptional-logit policy.

Dense target slices remain caller-obligated to be [finite](../../glossary.md#term-finite), non-negative, and normalized to one.
Construction reads no values and neither diagnoses nor renormalizes them. An exact zero target
[weight](../../glossary.md#term-weight) contributes [positive zero](../../glossary.md#nan-infinity-and-signed-zero) even when its [log](../../glossary.md#term-log) probability is [negative infinity](../../glossary.md#nan-infinity-and-signed-zero).
[BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16)/[FLOAT32](../../glossary.md#term-float32) dense results use FLOAT32 computation after floating promotion and
[FLOAT64](../../glossary.md#term-float64) results use FLOAT64. A [NaN](../../glossary.md#term-nan) or [positive infinity](../../glossary.md#nan-infinity-and-signed-zero) in a participating logits slice and an
all-negative-[infinity](../../glossary.md#term-infinity) slice produce NaN. With at least one finite logit, positive target weight on
negative infinity contributes positive infinity, while exact zero weight still contributes
positive zero.

An [index target](../../glossary.md#index-target) has exact type INT32 or INT64 and [Shape](../../glossary.md#shape) equal to the logits Shape with the class
axis removed. Target [axis](../../glossary.md#term-axis) `t` maps to logits axis `t` before the class axis and `t + 1` after it.
Unequal static mapped Dimensions fail; an unequal pair involving an unresolved Dimension leaves a
later equality obligation. No broadcasting, target promotion, one-hot conversion, negative-index
wrapping, or clamping occurs. `NONE` retains the exact target Shape object, including [scalar](../../glossary.md#term-scalar) Shape
for [rank-one](../../glossary.md#axis-rank-scalar-vector-and-matrix) logits. Index results retain exact logits type and use only logits
`requiresGrad` metadata.

For integer class target $y_g$, optional exact ignore value $I$, and class extent $C$:

$$
\ell_g=\begin{cases}
+0&I\text{ is present and }y_g=I,\\
a_g-z_{g,y_g}&\text{otherwise, requiring }0\leq y_g<C.
\end{cases}
$$

The group coordinate $g$ and log-sum-exp $a_g$ have the meanings above.
This is a conditional definition: the ignored branch does not evaluate the logits slice.

Exact ignore comparison occurs before bounds checking or logits [evaluation](../../glossary.md#term-evaluation), so the ignore value
may itself be out of range. Every non-matching target must eventually be in range. [Model](../../glossary.md#term-model)
construction reads no target or logits values—even from [eager](../../glossary.md#term-eager-evaluation) tensors—so bounds proof and safe
indexed access remain later [compiler](../../glossary.md#term-compiler), preparation, or execution obligations.

For logits `[1, 2, 3]`, `lse` is approximately `3.407605964`: index target `2` gives
`0.407605964`, index target `0` gives `2.407605964`, and INT32 target `-1` with exact
`ScalarValue.int32(-1)` gives positive zero without evaluating the logits slice. The same logits
with dense target `[0, 0, 1]` give `0.407605964`. These are mathematical examples, not values
computed while constructing a Tensor expression. The meanings are comparable to official
[PyTorch cross entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
and [ONNX SoftmaxCrossEntropyLoss](https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html)<span data-glossary-navigation="true"> ([ONNX glossary](../../glossary.md#term-onnx))</span>.

Let `S` be the target/[sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters)-domain size and `N` the non-ignored target count (`N = S` when no
ignore value is present):

| Reduction | Dense result | Index result | Result Shape |
|---|---|---|---|
| `LossReduction.NONE` | One weighted loss per group | One selected loss per target; ignored positions are `+0` | Class-axis-removed Shape; the index branch retains the exact target Shape |
| `LossReduction.SUM` | Sum over all groups | Sum over non-ignored targets | Shared `Shape.scalar()` |
| `LossReduction.MEAN` | Sum divided by `S` | Sum divided by `N` | Shared `Shape.scalar()` |

An empty domain produces empty `NONE`, positive-zero `SUM`, and NaN [`MEAN`](../../glossary.md#term-mean). An all-ignored,
non-empty index domain produces positive zeros for `NONE`, positive-zero `SUM`, and NaN `MEAN`.
Without ignore, a non-empty domain requires positive class [extent](../../glossary.md#term-extent). With ignore, zero class extent
is also valid when every target is ignored; because construction reads no values, it retains the
deferred alternative `S == 0 || classExtent > 0 || all targets equal ignoreIndex`. Every
non-ignored target is invalid when class extent is zero.

All four low types and FLOAT32 index computation uses FLOAT32 max, exponential, summation,
logarithm, subtraction, loss accumulation, and division; FLOAT64 uses FLOAT64. Final values round to logits
type. Equal-or-wider intermediates, compensation, vectorization, [fusion](../../glossary.md#term-fusion), and [reassociation](../../glossary.md#term-reassociation) remain
allowed. No narrower arithmetic, fixed traversal, bitwise [identity](../../glossary.md#term-identity-meanings), [NaN payload](../../glossary.md#nan-infinity-and-signed-zero)/sign, or identical
finite [rounding](../../glossary.md#term-rounding) across [backends](../../glossary.md#backend) is promised. For a non-ignored slice, any NaN or positive infinity
and an all-negative-infinity slice produce NaN. With at least one finite logit, selecting negative
infinity produces positive infinity; selecting a finite class gives its finite stable loss.
Ignored positions stay positive zero without propagating special values. Participating NaN
propagates through reductions; otherwise participating positive infinity produces positive
infinity.

Every successful request creates one fresh unlabeled, storage-free Tensor with unresolved [layout](../../glossary.md#layout),
one producer, one ID, and output-index-zero [provenance](../../glossary.md#provenance) over `[logits, target]`. It creates no
[softmax](../../glossary.md#softmax--log-softmax), [gather](../../glossary.md#gather), select, cast, arithmetic, reduction, or one-hot sub-producer and mutates no input.
These are current model-expression facts only. Package-private compiler [capture](../../glossary.md#term-capture-meanings) and
[descriptor](../../glossary.md#term-descriptor)-visible deferred proof are current. Compiler-owned first-order [autograd](../../glossary.md#autograd) supports
logits and target roles for dense floating targets. For index targets it supports logits only,
requires positive static class depth, clamps ignored targets before one-hot construction, and
excludes ignored rows through a final selection; dynamic or zero class depth fails closed.
Construction does not inspect index values or target contents. Value-dependent [validation](../../glossary.md#term-validation)
belongs to the accepted [backend route](../../glossary.md#backend-route), not to descriptor [inference](../../glossary.md#term-inference). Qualified occurrences
already lower, [prepare](../../glossary.md#prepare), execute and publish through [Engine](../../glossary.md#term-engine) and can supply configured Training
objectives; this adds no arbitrary dynamic class depth, implicit target [normalization](../../glossary.md#term-normalization) or
general-purpose optimization/[decomposition](../../glossary.md#term-decomposition) permission.


---

[Previous topic](general-scan.md) · [Next topic](reshape-expand.md)
