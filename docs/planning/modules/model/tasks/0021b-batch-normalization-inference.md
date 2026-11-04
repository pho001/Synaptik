# Task 0021B: [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[Normalization](../../../../glossary.md#term-normalization) [Inference](../../../../glossary.md#term-inference)

## Status

Complete

## Goal

Add one first-class [backend](../../../../glossary.md#backend)-independent `BATCH_NORM_INFERENCE` meaning and exactly one public
receiver expression. The [occurrence](../../../../glossary.md#term-occurrence) consumes explicit data, scale, [bias](../../../../glossary.md#term-bias), running [mean](../../../../glossary.md#term-mean), and
running [variance](../../../../glossary.md#term-variance) [tensors](../../../../glossary.md#tensor), applies stateless per-[channel](../../../../glossary.md#term-channel) [inference](../../../../glossary.md#term-inference) using an explicit normalized
channel [axis](../../../../glossary.md#term-axis) and exact typed [epsilon](../../../../glossary.md#epsilon), preserves the input [Shape](../../../../glossary.md#shape), and records exactly one output.

This task deliberately separates deterministic inference from every training [lifecycle](../../../../glossary.md#lifecycle). There is
no training/[evaluation](../../../../glossary.md#term-evaluation) flag, current-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [statistic](../../../../glossary.md#term-statistic) calculation, running-statistic update,
[momentum](../../../../glossary.md#term-momentum), saved output, mutation, [parameter](../../../../glossary.md#term-parameter) [owner](../../../../glossary.md#term-owner), or hidden state. All data required by the
formula is visible in the [producer](../../../../glossary.md#term-producer)'s ordered input list.

## [Scope](../../../../glossary.md#term-scope)

- Add `BatchNormKind.BATCH_NORM_INFERENCE` in the existing [normalization](../../../../glossary.md#term-normalization) [operation](../../../../glossary.md#operation) package.
- Add one [immutable](../../../../glossary.md#term-immutable) public `BatchNormInferenceAttrs(channelAxis, epsilon)` value containing only
  a normalized non-negative channel axis and exact typed epsilon.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [epsilon glossary](../../../../glossary.md#epsilon))</span>
- Give `BATCH_NORM_INFERENCE` exactly one fixed five-input/one-output [signature](../../../../glossary.md#term-signature).
- Add exactly one public receiver method and no static entry:

  ```java
  public Tensor batchNormInference(
          int channelAxis,
          Tensor scale,
          Tensor bias,
          Tensor runningMean,
          Tensor runningVariance,
          ScalarValue epsilon)
  ```

- Record exact ordered logical inputs
  `[input, scale, bias, runningMean, runningVariance]`; every operand is mandatory.
- Add one package-private stateless `TensorBatchNormInferenceExpressions` helper and exactly one
  final `TensorFactory.createDerived` delegation per successful request.
- Require input rank at least two, normalize one positive or negative caller channel axis, and
  allow that axis at any logical position without assuming NCHW, NHWC, or a resolved layout.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [NCHW glossary](../../../../glossary.md#term-nchw); [NHWC glossary](../../../../glossary.md#term-nhwc); [layout glossary](../../../../glossary.md#layout))</span>
- Require scale, bias, running mean, and running variance to be exact rank-one per-channel vectors,
  with local static mismatch rejection and explicit deferred symbolic equality obligations.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [vectors glossary](../../../../glossary.md#term-vector))</span>
- Require all five inputs to be floating, promote them in exact occurrence order, and require
  epsilon to have the exact result type.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Select the exact inference formula, running-variance interpretation, computation format, empty,
  negative-variance, NaN, infinity, signed-zero, overflow, reassociation, and determinism policies.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Produce one fresh unlabeled, storage-free, unresolved-layout result with exact input Shape,
  combined gradient eligibility, ordered provenance, and output index zero.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [provenance glossary](../../../../glossary.md#provenance))</span>
- Update focused Javadocs, Tensor and Compile API references, glossary, capability baseline,
  master plan, roadmap, task evidence, and public-surface tests in the same overall change.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- training-time batch mean or variance, reduction axes, statistic calculation, training mode,
  evaluation mode, mode flags, momentum, running-stat update formulas, mutation, saved mean,
  saved variance, saved inverse standard deviation, state inputs/outputs, or multi-output training<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [axes glossary](../../../../glossary.md#term-axis); [statistic glossary](../../../../glossary.md#term-statistic); [evaluation glossary](../../../../glossary.md#term-evaluation); [momentum glossary](../../../../glossary.md#term-momentum); [standard deviation glossary](../../../../glossary.md#term-standard-deviation))</span>
- any decision about task 0021C's operation kind, attributes, signatures, outputs, public surface,
  formulas, update policy, or implementation<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- optional affine operands, no-affine, scale-only, bias-only, implicit ones/zeros, nullable or
  optional operands, parameter initialization, layer/module objects, parameter ownership, or
  trainable registries<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [parameter glossary](../../../../glossary.md#term-parameter); [module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- scalar or broadcast affine/statistic operands, per-element or grouped parameters, multiple
  channel axes, inferred channel axis, a batch-axis attribute, NCHW/NHWC-specific aliases, or a
  general options object<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element); [channel glossary](../../../../glossary.md#term-channel); [NCHW glossary](../../../../glossary.md#term-nchw); [NHWC glossary](../../../../glossary.md#term-nhwc); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- rank-zero or rank-one data, default channel axis, default epsilon, untyped binary64 epsilon,
  epsilon Tensor input, configurable output type, or configurable computation/accumulator type<span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [epsilon glossary](../../../../glossary.md#epsilon); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [Tensor glossary](../../../../glossary.md#tensor); [accumulator type glossary](../../../../glossary.md#accumulator-type))</span>
- rejection by reading running-variance values, a non-negative-value proof object, variance clamp,
  absolute-value repair, standard-deviation input, inverse-variance input, correction conversion,
  or denominator sentinel
- integral, BOOL, FLOAT16, quantized, sparse, complex, or unsigned inputs; implicit or explicit
  cast-producer insertion<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16); [producer glossary](../../../../glossary.md#term-producer))</span>
- value reads, eager evaluation, host allocation, result storage, resolved layout, input mutation,
  materialization, interning, or canonicalization<span data-glossary-navigation="true"> ([eager evaluation glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [layout glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- algorithms, fixed traversal/evaluation order, fused multiply-add selection, gradients, adjoints,
  compiler capture, graph-wide constraint solving, binding, decomposition, fusion, backend
  support, lowering, prepare, runtime, execution, tolerances, conformance, or integration<span data-glossary-navigation="true"> ([fused multiply-add glossary](../../../../glossary.md#term-fused-multiply-add-fma); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [binding glossary](../../../../glossary.md#term-binding); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to layer/RMS normalization, softmax, reductions, typed scalar, Shape, signature,
  factory/producer/provenance foundations, architecture, dependencies, Gradle, other modules, or
  detailed specifications for later tasks<span data-glossary-navigation="true"> ([RMS normalization glossary](../../../../glossary.md#rms-normalization); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Public and [operation](../../../../glossary.md#operation) contracts

### Semantic kind, attributes, and [signature](../../../../glossary.md#term-signature)

`BatchNormKind` contains exactly `BATCH_NORM_INFERENCE` in this task. It does not extend or reuse
`LayerNormKind`, `RmsNormKind`, their attributes, or their trailing-[Shape](../../../../glossary.md#shape) semantics. The kind
accepts exactly:

| Attributes type | Ordered inputs | Ordered outputs |
|---|---|---|
| `BatchNormInferenceAttrs` | `[input, scale, bias, runningMean, runningVariance]` | `[output]` |

The family declares:

```java
OperationSignature.fixed(BatchNormInferenceAttrs.class, 5, 1)
```

No input range, optional position, second signature, mode attribute, training kind, or auxiliary
output is added. `BatchNormInferenceAttrs` contains exactly:

```java
int channelAxis
ScalarValue epsilon
```

Its compact constructor validates in component order: [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis), then [epsilon](../../../../glossary.md#epsilon). It requires a
non-negative [normalized axis](../../../../glossary.md#normalized-axis); requires non-null floating epsilon; and requires epsilon to be
[finite](../../../../glossary.md#term-finite) and strictly greater than [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) in its exact [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64)
representation. It retains the exact [immutable](../../../../glossary.md#term-immutable) epsilon reference. Exact intrinsic failures are:

```text
IllegalArgumentException("channelAxis must be non-negative: <channelAxis>")
NullPointerException("epsilon")
IllegalArgumentException("epsilon must have a floating data type, but was <dataType>")
IllegalArgumentException("epsilon must be finite and positive: <epsilon>")
```

Positive zero, [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero), negative finite values, either [infinity](../../../../glossary.md#term-infinity), and every [NaN](../../../../glossary.md#term-nan) encoding fail
the final epsilon check. The record stores no [Tensor](../../../../glossary.md#tensor), raw axis, input [rank](../../../../glossary.md#term-rank), channel [extent](../../../../glossary.md#term-extent),
operand-presence flag, result, [layout](../../../../glossary.md#layout), [gradient](../../../../glossary.md#term-gradient), [compiler](../../../../glossary.md#term-compiler), [backend](../../../../glossary.md#backend), [runtime](../../../../glossary.md#term-runtime), or training state.

### Public surface and ordered operand roles

The receiver is the data input. The five [producer](../../../../glossary.md#term-producer) positions have stable roles:

| Position | Role | Required [Shape](../../../../glossary.md#shape) |
|---|---|---|
| 0 | input data | [rank](../../../../glossary.md#term-rank) at least two, unchanged result Shape |
| 1 | scale | [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [vector](../../../../glossary.md#term-vector) `[C]` |
| 2 | [bias](../../../../glossary.md#term-bias) | rank-one vector `[C]` |
| 3 | running [mean](../../../../glossary.md#term-mean) | rank-one vector `[C]` |
| 4 | running [variance](../../../../glossary.md#term-variance) | rank-one vector `[C]` |

`C` is the input [extent](../../../../glossary.md#term-extent) at normalized `channelAxis`. Scale and bias are mandatory explicit affine
operands. Running mean and running variance are mandatory explicit estimated statistics. There is
no overload without any one of them, nullable value, `Optional<Tensor>`, hidden [constant](../../../../glossary.md#term-constant),
attribute-held affine/[statistic](../../../../glossary.md#term-statistic) value, attrs-taking public overload, static convenience, [alias](../../../../glossary.md#aliasing-and-arena),
or training/[evaluation](../../../../glossary.md#term-evaluation) switch.

The method constructs expression [metadata](../../../../glossary.md#term-metadata), not a layer. It creates, initializes, owns, registers,
or mutates no [parameter](../../../../glossary.md#term-parameter) or statistic and does not infer whether an input is trainable state.

### [Rank](../../../../glossary.md#term-rank), [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis), and [layout](../../../../glossary.md#layout)-neutral [Shape](../../../../glossary.md#shape) contract

Input rank `R` must satisfy `R >= 2`. The caller channel axis is normalized exactly once through
the current `Shape.normalizeAxis` rule, so valid raw values are in `[-R, R - 1]`. The normalized
non-negative value is retained in attributes. Every logical input axis is eligible to be the
channel axis; there is no distinguished [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-axis position and no [NCHW](../../../../glossary.md#term-nchw), [NHWC](../../../../glossary.md#term-nhwc), [NCW](../../../../glossary.md#term-ncw), NCL, or
[spatial](../../../../glossary.md#term-spatial-axes)-rank assumption. Every [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) sharing the same channel coordinate uses the same four
per-channel values.

The result retains the exact input Shape reference and every exact [Dimension](../../../../glossary.md#dimension) reference. Input
rank failure is:

```text
batchNormInference input rank must be at least 2, but was <rank>
```

An invalid raw channel axis then fails through the current Shape diagnostic:

```text
Axis <axis> is outside shape rank <rank>
```

No layout is inspected. A resolved input layout does not constrain or alter channel-axis meaning,
and the result layout is unresolved.

### [Parameter](../../../../glossary.md#term-parameter)/[statistic](../../../../glossary.md#term-statistic) [Shapes](../../../../glossary.md#shape) and symbolic [validation](../../../../glossary.md#term-validation)

Each of scale, [bias](../../../../glossary.md#term-bias), running [mean](../../../../glossary.md#term-mean), and running [variance](../../../../glossary.md#term-variance) must have [rank](../../../../glossary.md#term-rank) exactly one. Role checks
occur in that order. Exact rank failures are:

```text
batchNormInference scale rank must be one, but was <rank>
batchNormInference bias rank must be one, but was <rank>
batchNormInference runningMean rank must be one, but was <rank>
batchNormInference runningVariance rank must be one, but was <rank>
```

For each role, compare its sole [Dimension](../../../../glossary.md#dimension) to the input Dimension at normalized [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis):

- structurally equal static, dynamic, or expression Dimensions pass;
- unequal static Dimensions fail locally;
- any unequal pair involving an unresolved Dimension is accepted with an equality obligation for
  later compiler capture/binding because the result Shape remains exactly derivable; and<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>
- no parameter/statistic Shape or Dimension is copied, expanded, broadcast, or rewritten.

Static failures, checked in role order, are:

```text
batchNormInference scale channel dimension mismatch: input=<inputDimension>, scale=<scaleDimension>
batchNormInference bias channel dimension mismatch: input=<inputDimension>, bias=<biasDimension>
batchNormInference runningMean channel dimension mismatch: input=<inputDimension>, runningMean=<runningMeanDimension>
batchNormInference runningVariance channel dimension mismatch: input=<inputDimension>, runningVariance=<runningVarianceDimension>
```

The four [vectors](../../../../glossary.md#term-vector) need not contain the same exact Shape object. Equality to the one input channel
[extent](../../../../glossary.md#term-extent) is the semantic obligation. Ordinary [broadcasting](../../../../glossary.md#broadcasting), [scalar](../../../../glossary.md#term-scalar) operands, `[1, C]`, or a
full-rank [singleton](../../../../glossary.md#term-singleton)-expanded Shape are not accepted.

### Empty and [channel](../../../../glossary.md#term-channel)-zero behavior

A static channel [extent](../../../../glossary.md#term-extent) zero is valid only with four static `[0]` [vectors](../../../../glossary.md#term-vector). The result then has zero
[elements](../../../../glossary.md#term-element) and no formula is evaluated. An unresolved channel extent may later bind to zero if every
deferred vector equality also binds to zero. A zero on any non-channel [axis](../../../../glossary.md#term-axis) also makes the result
empty; the four vectors still describe the channel extent and are validated normally. Empty
results contain no normalized values, denominator, [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), or [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) result.

There is no division-by-count or [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[statistic](../../../../glossary.md#term-statistic) reduction in [inference](../../../../glossary.md#term-inference), so an empty non-channel
axis creates no special statistic rule. No [eager](../../../../glossary.md#term-eager-evaluation) storage inspection is used to discover emptiness
beyond static [Shape](../../../../glossary.md#shape) [metadata](../../../../glossary.md#term-metadata).

### Formula, [epsilon](../../../../glossary.md#epsilon) placement, and running [variance](../../../../glossary.md#term-variance)

For an output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) whose [channel](../../../../glossary.md#term-channel) is `c`, define:

```text
centered     = input - runningMean[c]
denominator  = sqrt(runningVariance[c] + epsilon)
standardized = centered / denominator
output       = standardized * scale[c] + bias[c]
```

Epsilon is added to running variance inside the [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions). Affine order is multiply by scale,
then add [bias](../../../../glossary.md#term-bias). The supplied running variance is interpreted as an estimated per-channel variance
and used directly. It is not a [standard deviation](../../../../glossary.md#term-standard-deviation), inverse variance, inverse standard deviation,
sum of squares, or correction-bearing [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) [statistic](../../../../glossary.md#term-statistic). [Inference](../../../../glossary.md#term-inference) performs no population/sample
correction conversion, recomputation from input, [momentum](../../../../glossary.md#term-momentum) update, or mutation.

For [FLOAT64](../../../../glossary.md#term-float64) data `[1, 2]` in one channel with scale `2`, bias `0.5`, running [mean](../../../../glossary.md#term-mean) `1`, running
variance `4`, and epsilon `1e-5`, the outputs are approximately
`[0.5, 1.4999987500023437]`. This is a mathematical illustration only; construction reads no
values and computes no result.

This explicit five-input, one-output inference formula follows official
[ONNX BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html).
Synaptik deliberately uses an explicit [layout](../../../../glossary.md#layout)-neutral channel [axis](../../../../glossary.md#term-axis), exact [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [vectors](../../../../glossary.md#term-vector), and
exact typed epsilon rather than [ONNX](../../../../glossary.md#term-onnx)'s fixed channel position, defaults, or combined mode surface.

### Negative running [variance](../../../../glossary.md#term-variance)

[Model](../../../../glossary.md#term-model) construction does not read or prove running-variance values and therefore accepts a [Tensor](../../../../glossary.md#tensor)
whose eventual values may be negative. There is no clamp, [absolute value](../../../../glossary.md#norms-elementary-functions-and-probability-distributions), [eager](../../../../glossary.md#term-eager-evaluation) rejection, hidden
[validation](../../../../glossary.md#term-validation) output, or replacement denominator. The formula is authoritative:

- [finite](../../../../glossary.md#term-finite) `runningVariance[c] + epsilon < 0` produces a [NaN](../../../../glossary.md#term-nan) denominator and NaN output;
- an exact zero sum produces a zero denominator, so centered zero yields NaN and non-zero centered
  values yield signed infinity before affine operations;<span data-glossary-navigation="true"> ([infinity glossary](../../../../glossary.md#term-infinity))</span>
- a negative variance greater than `-epsilon` leaves a positive radicand and follows the ordinary
  formula; and
- [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) produces NaN through the [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions).

This policy does not claim that negative variance is a valid trained [statistic](../../../../glossary.md#term-statistic). It defines
observable formula behavior without adding value-reading validation to model construction.

### Data types, promotion, and computation format

Input, scale, [bias](../../../../glossary.md#term-bias), running [mean](../../../../glossary.md#term-mean), and running [variance](../../../../glossary.md#term-variance) must each be [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT32](../../../../glossary.md#term-float32), or [FLOAT64](../../../../glossary.md#term-float64).
Derive result type by applying `DataTypePromotion.promoteFloating` in exact [producer](../../../../glossary.md#term-producer) [occurrence](../../../../glossary.md#term-occurrence)
order: input with scale, then bias, then running mean, then running variance. Promotion changes
result [metadata](../../../../glossary.md#term-metadata) only and inserts no cast producer.

Exact type failures are:

```text
batchNormInference input must have a floating data type, but was <dataType>
batchNormInference scale must have a floating data type, but was <dataType>
batchNormInference bias must have a floating data type, but was <dataType>
batchNormInference runningMean must have a floating data type, but was <dataType>
batchNormInference runningVariance must have a floating data type, but was <dataType>
batchNormInference epsilon data type must match result data type: epsilon=<epsilonType>, result=<resultType>
```

[Epsilon](../../../../glossary.md#epsilon) must exactly equal the promoted result type. There is no reduction [accumulator](../../../../glossary.md#accumulator-type) in
[inference](../../../../glossary.md#term-inference). Formula arithmetic treats all five inputs and epsilon in FLOAT32 when the result is
BFLOAT16 or FLOAT32, and in FLOAT64 when the result is FLOAT64; the final value is rounded to the
result format. No narrower computation format or independently configurable accumulator/result
type is allowed.

For [finite](../../../../glossary.md#term-finite) values, the semantic [target](../../../../glossary.md#term-target) is the stated real formula rounded to the result format.
A conforming implementation may use equal-or-wider intermediates, reassociate operations, or use
[fused multiply-add](../../../../glossary.md#term-fused-multiply-add-fma) only when later [conformance](../../../../glossary.md#term-conformance) tolerances and the special-value classes below are
preserved. [Model](../../../../glossary.md#term-model) selects no algorithm. Bitwise equality, one fixed arithmetic grouping, [NaN
payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign preservation, and cross-[backend](../../../../glossary.md#backend) identical finite [rounding](../../../../glossary.md#term-rounding) are not promised.

### [NaN](../../../../glossary.md#term-nan), [infinity](../../../../glossary.md#term-infinity), [signed zero](../../../../glossary.md#term-signed-zero), [overflow](../../../../glossary.md#term-overflow), and determinism

- A NaN participating at a coordinate in input, scale, bias, running mean, running variance, or
  epsilon-derived arithmetic propagates to NaN output. No later affine zero suppresses it.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [bias glossary](../../../../glossary.md#term-bias); [mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [epsilon glossary](../../../../glossary.md#epsilon))</span>
- Positive-infinite running variance produces positive-infinite denominator. A finite centered
  numerator produces signed zero before affine multiplication. Infinite centered numerator divided
  by infinite denominator produces NaN.
- Positive or negative infinite input/running mean follows ordinary subtraction: same-signed
  infinity subtraction is NaN, while a sole infinity yields the corresponding signed infinity.
  Division and affine operations then follow ordinary floating rules.
- Negative-infinite running variance and every negative radicand produce NaN. Positive-infinite
  scale times standardized zero produces NaN; infinite scale times non-zero finite standardized
  value produces signed infinity. A NaN bias produces NaN; a sole infinite bias determines the
  final infinity unless the preceding product is the opposing infinity, which produces NaN.
- With finite operands and positive denominator, centered and standardized zero signs follow
  ordinary subtraction and division. Multiplication and final addition follow ordinary floating
  signed-zero rules. No canonical zero sign is imposed.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Finite subtraction, addition, multiplication, or division may overflow or underflow in the
  selected computation format and then follows its ordinary infinity, zero, and NaN rules.<span data-glossary-navigation="true"> ([underflow glossary](../../../../glossary.md#term-underflow))</span>
- Empty results contain no value-level special cases.

Because [inference](../../../../glossary.md#term-inference) is coordinatewise and performs no reduction, there is no traversal-order
[dependency](../../../../glossary.md#term-dependency). Separately constructed equal requests remain distinct [producers](../../../../glossary.md#term-producer). [Finite](../../../../glossary.md#term-finite) [rounding](../../../../glossary.md#term-rounding) may
still differ under permitted grouping or [fused](../../../../glossary.md#specialization-and-fusion) operations, so bitwise cross-[backend](../../../../glossary.md#backend) determinism is
not promised; the special-value classes above are deterministic semantic constraints.

### Result [metadata](../../../../glossary.md#term-metadata), [provenance](../../../../glossary.md#provenance), and [identity](../../../../glossary.md#term-identity-meanings) effects

The [descriptor](../../../../glossary.md#term-descriptor) uses the ordered-promoted result type, exact input [Shape](../../../../glossary.md#shape) reference, unresolved
[layout](../../../../glossary.md#layout), and `requiresGrad` equal to the logical OR of all five input descriptors' eligibility.
Every result is fresh, unlabeled, and storage-free. Inputs and their descriptors, IDs, labels,
storage, layout, [gradient](../../../../glossary.md#term-gradient) eligibility, and provenance are not mutated.

Each success creates one [`Operation`](../../../../glossary.md#operation), one [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), one output descriptor, one [Tensor](../../../../glossary.md#tensor)
wrapper, and one Tensor ID. Provenance output index is zero and exact ordered inputs are
`[input, scale, bias, runningMean, runningVariance]`. No hidden [constant](../../../../glossary.md#term-constant), [saved statistic](../../../../glossary.md#saved-statistic), sibling
descriptor, additional wrapper, state [token](../../../../glossary.md#term-token), or extra ID is created. Future [compiler](../../../../glossary.md#term-compiler) or training
work cannot retroactively change this [inference](../../../../glossary.md#term-inference) [producer](../../../../glossary.md#term-producer)'s fixed one-output contract.

### [Validation](../../../../glossary.md#term-validation) and construction order

`TensorBatchNormInferenceExpressions` exposes exactly one package-private entry matching the
public receiver method. It validates in this order:

1. null-check `input`, `scale`, [`bias`](../../../../glossary.md#term-bias), `runningMean`, `runningVariance`, then [`epsilon`](../../../../glossary.md#epsilon);
2. validate floating eligibility for input, scale, bias, running [mean](../../../../glossary.md#term-mean), then running [variance](../../../../glossary.md#term-variance);
3. require input [rank](../../../../glossary.md#term-rank) at least two;
4. normalize the caller [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis) exactly once through `Shape.normalizeAxis`;
5. require rank one for scale, bias, running mean, then running variance;
6. validate static/symbolic channel compatibility for those four roles in the same order;
7. promote the five input data types in exact [occurrence](../../../../glossary.md#term-occurrence) order;
8. validate epsilon exact result type;
9. construct `BatchNormInferenceAttrs`, [descriptor](../../../../glossary.md#term-descriptor), [operation](../../../../glossary.md#operation), and delegate exactly once.

Null failures use exact [parameter](../../../../glossary.md#term-parameter) names. The attributes constructor defensively repeats its
intrinsic normalized-axis and epsilon validation. Existing [Shape](../../../../glossary.md#shape), promotion, descriptor,
operation, [producer](../../../../glossary.md#term-producer), and identifier failures retain their current messages. Every local failure
occurs before factory delegation and consumes no [Tensor](../../../../glossary.md#tensor) ID, producer, descriptor wrapper, or
result Tensor.

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
- [Layer normalization](0021-layer-normalization-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([Layer normalization glossary](../../../../glossary.md#layer-normalization))</span>
- [RMS normalization](0021a-rms-normalization-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([RMS normalization glossary](../../../../glossary.md#rms-normalization))</span>
- [Operation signatures](0018k-operation-signature-and-construction-hardening.md)<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- [Typed scalar values](0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>

## Architecture constraints

- Work stays in [model](../../../../glossary.md#term-model) plus its documentation/[planning](../../../../glossary.md#term-planning). [`Tensor`](../../../../glossary.md#tensor) remains public mutable API state,
  not graph intermediate representation.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [intermediate representation glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Batch-normalization inference operation types contain backend-independent semantic parameters
  only and do not import Tensor, graph, compiler, training, runtime, prepare, or backend types.<span data-glossary-navigation="true"> ([Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [inference glossary](../../../../glossary.md#term-inference); [backend glossary](../../../../glossary.md#backend); [parameters glossary](../../../../glossary.md#term-parameter); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare))</span>
- Package direction is tensor helper to normalization operation/datatype/shape. No reverse edge,
  registry, service, layer owner, parameter owner, statistic owner, or state owner is introduced.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [registry glossary](../../../../glossary.md#term-registry); [owner glossary](../../../../glossary.md#term-owner); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- Explicit running-statistic Tensors are immutable logical inputs to this occurrence. The
  operation does not own or mutate them and has no hidden training/evaluation lifecycle.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [occurrence glossary](../../../../glossary.md#term-occurrence); [evaluation glossary](../../../../glossary.md#term-evaluation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Compiler owns capture, deferred equality proof, binding validation, gradients, saved values,
  adjoints, and legal decomposition. Backend prepare owns algorithms, lowering, specialization,
  fusion, kernels, and tolerance satisfaction. Runtime executes prepared work without consuming<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization); [fusion glossary](../../../../glossary.md#term-fusion); [kernels glossary](../../../../glossary.md#kernel); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
  [`Operation`](../../../../glossary.md#operation) on its [hot path](../../../../glossary.md#term-hot-path).
- No architecture, focused-architecture, dependency, lifecycle, training, Gradle, or cross-module
  change is authorized.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.normalization` — add inference-only batch-
  normalization semantics beside unchanged softmax, layer-normalization, and RMS-normalization
  families.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Packages added: None.

Type placement:

- `...operation.normalization.BatchNormKind` — public inference semantic identity and fixed
  occurrence signature.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [signature glossary](../../../../glossary.md#term-signature))</span>
- `...operation.normalization.BatchNormInferenceAttrs` — public normalized channel-axis and exact
  typed epsilon parameters.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [epsilon glossary](../../../../glossary.md#epsilon); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- `...tensor.TensorBatchNormInferenceExpressions` — package-private validation, promotion, Shape,
  descriptor, and one-output provenance construction owner.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [owner glossary](../../../../glossary.md#term-owner))</span>
- `...tensor.Tensor` — established public fluent receiver [facade](../../../../glossary.md#term-facade).

Tests mirror production packages where package-private access is required.

## Affected files

Production (4):

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/BatchNormKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/normalization/BatchNormInferenceAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests (9):

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/normalization/BatchNormInferenceSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java` — exact
  signature and public count 183 to 184.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
  — count only, 183 to 184.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
  — count only, 183 to 184.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
  — count only, 183 to 184.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
  — count only, 183 to 184.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
  — count only, 183 to 184.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
  — count only, 183 to 184; explicitly authorized as the sole path-20 addition after the first
  full model run discovered its pre-existing exact-count assertion.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [run glossary](../../../../glossary.md#term-run))</span>

Documentation/[planning](../../../../glossary.md#term-planning) (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: [Runtime](../../../../glossary.md#term-runtime) and Training APIs; layer/[RMS normalization](../../../../glossary.md#rms-normalization), [softmax](../../../../glossary.md#softmax--log-softmax),
reductions/statistics, typed [scalar](../../../../glossary.md#term-scalar), [Shape](../../../../glossary.md#shape), [operation](../../../../glossary.md#operation)/[signature](../../../../glossary.md#term-signature), [Tensor](../../../../glossary.md#tensor)/factory/[producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance),
and other operation contracts; architecture/ADRs/tests; [conformance](../../../../glossary.md#term-conformance)/integration; [Gradle](../../../../glossary.md#term-gradle); other
[modules](../../../../glossary.md#term-module).

## Maximum [scope](../../../../glossary.md#term-scope)

Exactly 20 paths maximum: four production, nine test, and seven documentation/[planning](../../../../glossary.md#term-planning) paths.
The original task was one path above the planning guide's normal 18-path upper guardrail because the one new
public method requires six established exact public-count assertions to advance together, while
the semantic kind, typed attrs, helper, focused semantics/expression tests, and mandatory seven-
path documentation [handoff](../../../../glossary.md#term-handoff) are inseparable for an actionable public capability. The user
explicitly authorized that cohesive 19-path scope. During the first full [model](../../../../glossary.md#term-model) [run](../../../../glossary.md#term-run), the
pre-existing `TensorRmsNormExpressionTest` independently asserted the same public count and failed
only because it still expected 183. The user then explicitly authorized that necessary count-only
update as the sole path-20 addition. Stop for path 21, another production type,
test, document, helper change, architecture/[Gradle](../../../../glossary.md#term-gradle)/cross-[module](../../../../glossary.md#term-module) work, optional affine form,
training/state behavior, multi-output behavior, or a later detailed task specification.

## [Javadoc](../../../../glossary.md#term-javadoc) and documentation requirements

- Fully document kind, attrs, helper, and Tensor method: five ordered roles, layout-neutral channel
  axis, exact vector Shapes, formula/running-variance/epsilon contract, promotion/computation,
  empty/negative/special values, metadata/provenance/ID effects, validation/failures, and lifecycle
  boundaries.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [layout glossary](../../../../glossary.md#layout); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [vector glossary](../../../../glossary.md#term-vector); [Shapes glossary](../../../../glossary.md#shape); [variance glossary](../../../../glossary.md#term-variance); [epsilon glossary](../../../../glossary.md#epsilon); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Every input, non-void result, and expected failure has complete `@param`, `@return`, and
  `@throws` documentation.
- Tensor API gets the exact signature, operand/Shape table, one finite numerical example, static
  mismatch, symbolic deferral, empty/channel-zero and negative/special-value policies, provenance,
  and current-model versus planned compiler/execution boundary.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [signature glossary](../../../../glossary.md#term-signature); [finite glossary](../../../../glossary.md#term-finite); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Compile API records batch-normalization inference metadata as current after implementation and
  leaves capture, deferred constraint proof, gradients, saved values, decomposition, lowering,
  training transition, and execution planned.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [inference glossary](../../../../glossary.md#term-inference); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [lowering glossary](../../../../glossary.md#lowering))</span>
- Review glossary terms batch normalization, batch-normalization inference, channel axis, running
  mean, running variance, estimated statistic, epsilon, affine transform, computation format,
  producer, and saved statistic; add/refine only reusable distinctions needed by the API.<span data-glossary-navigation="true"> ([batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [mean glossary](../../../../glossary.md#term-mean); [statistic glossary](../../../../glossary.md#term-statistic); [affine transform glossary](../../../../glossary.md#affine-transform); [producer glossary](../../../../glossary.md#term-producer); [saved statistic glossary](../../../../glossary.md#saved-statistic))</span>
- Synchronize statuses: 0021 and 0021A remain Complete; 0021B is Ready during implementation and
  becomes Complete only after every criterion passes; 0021C and 0022–0024 remain Draft without
  detailed specifications.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>

## Acceptance criteria

- Exact `BATCH_NORM_INFERENCE`, one fixed attrs/signature variant, and one canonical receiver method
  exist; existing normalization families remain unchanged and public Tensor count is 184.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [canonical glossary](../../../../glossary.md#term-canonical-form); [normalization glossary](../../../../glossary.md#term-normalization); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Exact mandatory ordered roles and one output are locked; no optional/hidden operand, static
  entry, alias, attrs overload, mode flag, saved output, or mutation exists.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Rank-at-least-two input, one normalized layout-neutral channel axis, exact rank-one parameter/
  statistic Shapes, static mismatch rejection, symbolic equality deferral, exact input Shape
  retention, and empty/channel-zero rules match this task.<span data-glossary-navigation="true"> ([Rank glossary](../../../../glossary.md#term-rank); [layout glossary](../../../../glossary.md#layout); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [parameter glossary](../../../../glossary.md#term-parameter); [statistic glossary](../../../../glossary.md#term-statistic); [Shapes glossary](../../../../glossary.md#shape))</span>
- Ordered five-input floating promotion, exact-result-typed finite positive epsilon, and selected
  FLOAT32/FLOAT64 computation formats pass without cast insertion or value evaluation.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [epsilon glossary](../../../../glossary.md#epsilon); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Exact inference formula, direct estimated-variance interpretation, epsilon placement, negative
  variance, NaN/infinity/signed-zero, overflow, reassociation, rounding, and determinism policies
  are explicit and internally consistent.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [variance glossary](../../../../glossary.md#term-variance); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [reassociation glossary](../../../../glossary.md#term-reassociation); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Requires-grad OR, unresolved layout, freshness, no label/storage/mutation, exact input order, one
  output/index zero, and exact one-ID/producer/wrapper effects pass.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- [Validation](../../../../glossary.md#term-validation) order and exact task-owned messages pass; every pre-factory failure consumes no ID.
- No training statistic/update/momentum/multi-output detail, gradient/compiler/algorithm/backend/
  runtime behavior, architecture/dependency/build/cross-module change, or later spec is added.<span data-glossary-navigation="true"> ([momentum glossary](../../../../glossary.md#term-momentum); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- Focused tests, the recorded discovery model run and one final stable model suite,
  Javadoc/docs/link/anchor/fence/newline/whitespace, official-link, exact 20-path/package/public-
  surface/status audits, and <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes all affected
  Javadocs/docs/glossary impact, and records reasoned no-change conclusions before completion.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormInferenceExpressionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.datatype.ScalarValueTest --tests io.github.pho001.synaptik.model.shape.ShapeTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Focused tests cover exact kind/attrs/[signature](../../../../glossary.md#term-signature), public surface, ordered operand roles, attrs
validation, [axis](../../../../glossary.md#term-axis) [normalization](../../../../glossary.md#term-normalization), [rank](../../../../glossary.md#term-rank)/static/symbolic/empty [Shapes](../../../../glossary.md#shape), promotion/computation and
formula policies as inspectable semantic contracts, negative/special values without [eager](../../../../glossary.md#term-eager-evaluation)
execution, [metadata](../../../../glossary.md#term-metadata), validation/ID effects, freshness, and exact one-output [provenance](../../../../glossary.md#provenance).

Documentation pass after final [Javadoc](../../../../glossary.md#term-javadoc):

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation-focused pass also runs targeted Markdown link/anchor/fence/final-newline and
trailing-whitespace checks, verifies the official [ONNX](../../../../glossary.md#term-onnx) link, audits exact 20-path [scope](../../../../glossary.md#term-scope)/package
placement, checks public count/signature and synchronized statuses, and records reused Java
evidence. Repository-wide validation is deferred to the selected modern-[operation](../../../../glossary.md#operation) [checkpoint](../../../../glossary.md#term-checkpoint)
after task 0022 and [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes one [module](../../../../glossary.md#term-module) without dependencies, architecture,
shared build configuration, or executable [backend](../../../../glossary.md#backend) behavior.

## Dependencies

- 0001–0002: [DataType](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [Dimension](../../../../glossary.md#dimension), static/symbolic, [rank](../../../../glossary.md#term-rank), and [axis](../../../../glossary.md#term-axis)-[normalization](../../../../glossary.md#term-normalization) contracts.
- 0005–0007 and 0011–0013: [operation](../../../../glossary.md#operation), [descriptor](../../../../glossary.md#term-descriptor), [Tensor](../../../../glossary.md#tensor), factory, and [provenance](../../../../glossary.md#provenance) foundations.
- 0018K: exact attributes-class [signatures](../../../../glossary.md#term-signature) and [occurrence](../../../../glossary.md#term-occurrence) [cardinality](../../../../glossary.md#term-cardinality).
- 0018L: shared [producer](../../../../glossary.md#term-producer)/output-index provenance, used here in one-output form.
- 0018N: exact typed [scalar](../../../../glossary.md#term-scalar) representation for [epsilon](../../../../glossary.md#epsilon).
- 0018V: variance, correction, computation/accumulation, empty, special-value, and determinism
  vocabulary precedents; inference consumes an explicit variance rather than reducing input.<span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance); [inference glossary](../../../../glossary.md#term-inference))</span>
- 0021–0021A: typed epsilon, normalization metadata, one-output provenance, documentation, and
  lifecycle-separation precedents without reusing their trailing-Shape contracts.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

All dependencies are Complete.

## Follow-up tasks

- 0021C (Draft) remains the separately bounded training/statistic-transition frontier. This task
  establishes only that any future training work must not mutate or hide state and must not alter
  the completed one-output inference occurrence. No 0021C signature, formula, output, momentum,
  saved-statistic, or public-surface decision is made here.<span data-glossary-navigation="true"> ([statistic glossary](../../../../glossary.md#term-statistic); [inference glossary](../../../../glossary.md#term-inference); [occurrence glossary](../../../../glossary.md#term-occurrence); [signature glossary](../../../../glossary.md#term-signature); [momentum glossary](../../../../glossary.md#term-momentum))</span>
- 0022–0024 remain Draft at their established IDs. Task 0023 owns any batch-normalization adjoint
  or compiler-generated saved-statistic meaning justified by compiler implementation.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [adjoint glossary](../../../../glossary.md#term-adjoint); [compiler glossary](../../../../glossary.md#term-compiler))</span>

## Architecture impact

Expected impact: None.

The task adds [model](../../../../glossary.md#term-model)-owned [inference](../../../../glossary.md#term-inference) semantics and public expression [metadata](../../../../glossary.md#term-metadata) inside existing
package and [dependency](../../../../glossary.md#term-dependency) boundaries. If implementation requires hidden mode/state, mutation, a
cross-[module](../../../../glossary.md#term-module) type, architecture/dependency change, or [compiler](../../../../glossary.md#term-compiler)/[backend](../../../../glossary.md#backend)/[runtime](../../../../glossary.md#term-runtime) behavior, stop and
report the conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, and
docs/planning/modules/model/tasks/0021b-batch-normalization-inference.md. Implement that task
exactly as specified and do not implement out-of-scope training, state-transition, or later work.
Stop and report any architecture, dependency, or maximum-scope conflict. Do not commit or push
unless separately authorized.

After Java implementation and recorded module-test evidence, hand the final diff and evidence to
a separate documentation-focused agent/thread with clean context. That pass must follow
docs/developer-guide/documentation-rules.md, inspect source/tests, finalize Javadocs, Tensor and
Compile APIs, glossary and planning records, run Javadoc/Markdown/scope/status/whitespace checks,
and reuse successful Java evidence unless executable Java changes.

Update this task's local decisions, limitations, evidence, notes, completion summary, and status.
Do not mark Complete before the documentation pass and all acceptance criteria finish.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Provide the clean documentation context with this task, the complete implementation diff, exact
focused/final Java commands and results, affected [inference](../../../../glossary.md#term-inference) semantics/public method, architecture
and 20-path constraints, expected [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) and glossary/[planning](../../../../glossary.md#term-planning) impact, the official
[ONNX](../../../../glossary.md#term-onnx) link, and required [Javadoc](../../../../glossary.md#term-javadoc)/Markdown/[scope](../../../../glossary.md#term-scope)/status/whitespace evidence. It must inspect the
actual source/tests, not rely only on the handoff summary, and record reasoned no-change
conclusions.

## Local decisions

- Keep `BatchNormInferenceAttrs` limited to normalized channel axis and exact typed epsilon;
  occurrence-specific Tensor roles, rank, Shape, promotion, and result facts remain at the public
  expression boundary.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [epsilon glossary](../../../../glossary.md#epsilon); [occurrence glossary](../../../../glossary.md#term-occurrence); [Tensor glossary](../../../../glossary.md#tensor); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape))</span>
- Use direct running variance with epsilon inside the square root and exact ordered five-input
  promotion. Negative running values are not read during construction and receive no repair.<span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance); [square root glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>
- The first full [model](../../../../glossary.md#term-model) [run](../../../../glossary.md#term-run) exposed one additional pre-existing exact public-count assertion in
  `TensorRmsNormExpressionTest`. The user authorized only its 183-to-184 update as path 20; no
  other scope expansion was made.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Known limitations

- This task constructs model metadata only. Compiler capture, deferred equality proof, gradients,
  saved values, decomposition, backend preparation, numerical execution, and conformance remain
  unimplemented in their owning layers.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [decomposition glossary](../../../../glossary.md#term-decomposition); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Training-time batch statistics, running-statistic transition, momentum, and multi-output saved
  statistics remain Draft task 0021C without a detailed specification.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [statistic glossary](../../../../glossary.md#term-statistic); [momentum glossary](../../../../glossary.md#term-momentum))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/task_0021b_implementation` ran the exact focused command listed
  in this task; it passed `BUILD SUCCESSFUL in 8s`. Its first full
  `./gradlew :modules:model:test` discovered only the pre-existing 183 public-count expectation in
  `TensorRmsNormExpressionTest`. After the explicitly authorized count-only 183-to-184 correction,
  the isolated test passed `BUILD SUCCESSFUL in 1s`. The final stable
  `./gradlew :modules:model:test` passed `BUILD SUCCESSFUL in 1s` with 919 tests. The clean
  documentation pass changed no executable Java afterward and did not repeat those suites.
- Clean documentation context `/root/task_0021b_implementation/docs_finalize` applied the General,
  API/Javadoc, Planning, glossary, and example profiles. It independently reviewed the four
  production contracts, nine tests, Tensor/Compile/Runtime/Training APIs, glossary, capabilities,
  task, master plan, roadmap, architecture, and directly related normalization, scalar, Shape,
  signature, factory, producer, and provenance contracts. Existing production Javadocs were
  accurate and needed no further edit.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [normalization glossary](../../../../glossary.md#term-normalization); [scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape); [signature glossary](../../../../glossary.md#term-signature); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- `./gradlew :modules:model:javadoc` passed `BUILD SUCCESSFUL in 1s` after the final [Javadoc](../../../../glossary.md#term-javadoc) review.
- The targeted Markdown checker resolved 617 local links, including 166 heading anchors, across
  all seven documentation/planning paths with zero failures. Separate checks found balanced
  backtick and tilde fences, final newlines, no trailing whitespace, and zero failures. Manual
  recalculation confirmed the FLOAT64 <span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64))</span>`[1, 2]` example results `0.5` and approximately
  `1.4999987500023437`.
- `curl -I -L --max-time 20 https://onnx.ai/onnx/operators/onnx__BatchNormalization.html`
  returned HTTP 200 for the official ONNX BatchNormalization reference.<span data-glossary-navigation="true"> ([ONNX glossary](../../../../glossary.md#term-onnx))</span>
- `javap -public` inspection found exactly 184 public [`Tensor`](../../../../glossary.md#tensor) methods and exactly one
  `batchNormInference(int, Tensor, Tensor, Tensor, Tensor, ScalarValue)` receiver returning
  `Tensor`. Source/test inspection confirmed the sole `BATCH_NORM_INFERENCE` enum value, exact
  fixed five-input/one-output signature, one package-private helper entry, ordered provenance,
  and no static public entry, alias, optional operand, training flag, or auxiliary output.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Final inventory found exactly 20 authorized paths: four production, nine tests, and seven
  documentation/planning paths. Production types remain in the planned normalization/tensor
  packages with model-only imports; no architecture, dependency, Gradle, cross-module, later-task
  specification, training, compiler, backend, runtime, conformance, or integration path changed.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [module glossary](../../../../glossary.md#term-module); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Runtime API remains accurate unchanged because no prepared execution, storage, state, or run
  behavior was added. Training API remains accurate unchanged because no parameter, optimizer,
  statistic transition, momentum, saved output, or training mode was added. Related operation,
  layer/RMS normalization, scalar, Shape, signature, factory, producer, and provenance contracts
  remain accurate unchanged because the new family composes their established boundaries without
  modifying them. Architecture/ADRs/tests, conformance/integration, Gradle, and other modules need
  no change because ownership, dependencies, build behavior, backend behavior, and end-to-end
  execution did not change.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [run glossary](../../../../glossary.md#term-run); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [statistic glossary](../../../../glossary.md#term-statistic); [momentum glossary](../../../../glossary.md#term-momentum); [operation glossary](../../../../glossary.md#operation); [RMS normalization glossary](../../../../glossary.md#rms-normalization); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Final status audit confirmed tasks 0021, 0021A, and 0021B Complete; task 0021C and tasks
  0022–0024 Draft without detailed specifications. Final fence/newline/whitespace and
  `git diff --check` validation passed.

## Implementation notes

- Added `BatchNormKind`, `BatchNormInferenceAttrs`, one package-private construction helper, and
  one canonical public receiver. The helper validates all five floating operands, rank and
  channel-vector relationships, ordered promotion, exact epsilon, result metadata, freshness,
  and one-output provenance before its sole factory delegation.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [rank glossary](../../../../glossary.md#term-rank); [channel glossary](../../../../glossary.md#term-channel); [vector glossary](../../../../glossary.md#term-vector); [epsilon glossary](../../../../glossary.md#epsilon); [metadata glossary](../../../../glossary.md#term-metadata); [provenance glossary](../../../../glossary.md#provenance))</span>
- Finalized Tensor and Compile API references, glossary terminology, capability baseline, task,
  master plan, and roadmap. Current model behavior is separated explicitly from planned compiler,
  training, backend, runtime, and execution work.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Completion summary

- Completed changes: stateless explicit five-input batch-normalization inference semantics,
  attributes, public expression construction, tests, Javadocs, and synchronized documentation.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [inference glossary](../../../../glossary.md#term-inference))</span>
- Files changed or created: exactly 20 authorized paths — four production, nine tests, and seven
  documentation/planning files listed above.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: focused suite, authorized isolated count correction, final 919-test model
  suite, model Javadoc, 617-link/166-anchor Markdown, official ONNX link, public surface, exact
  scope/package/status, fence/newline/trailing-whitespace, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [ONNX glossary](../../../../glossary.md#term-onnx); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` all passed.
- Documentation-agent review: completed in clean context
  `/root/task_0021b_implementation/docs_finalize` using the required profiles and reused Java
  evidence without rerunning successful executable suites.
- Documentation impact: Tensor API, Compile API, glossary, capabilities, task, master plan, and
  roadmap finalized; Runtime and Training APIs and related architecture/docs remain accurate
  unchanged for the reasons recorded above.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: all four production contracts are complete and consistent with source/tests;
  generated model Javadoc passed without requiring another Java-source edit.
- Glossary impact: added the reusable batch-normalization inference, channel-axis, running-
  statistic, direct-variance, computation-format, epsilon, affine, and saved-statistic
  distinctions.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [statistic glossary](../../../../glossary.md#term-statistic); [variance glossary](../../../../glossary.md#term-variance); [epsilon glossary](../../../../glossary.md#epsilon))</span>
- Unresolved issues: None.
- Follow-up required: None for this task; Draft task 0021C remains separately planned work.

Status: Complete
