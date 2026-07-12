# Task 0021B: Batch-Normalization Inference

## Status

Complete

## Goal

Add one first-class backend-independent `BATCH_NORM_INFERENCE` meaning and exactly one public
receiver expression. The occurrence consumes explicit data, scale, bias, running mean, and
running variance tensors, applies stateless per-channel inference using an explicit normalized
channel axis and exact typed epsilon, preserves the input Shape, and records exactly one output.

This task deliberately separates deterministic inference from every training lifecycle. There is
no training/evaluation flag, current-batch statistic calculation, running-statistic update,
momentum, saved output, mutation, parameter owner, or hidden state. All data required by the
formula is visible in the producer's ordered input list.

## Scope

- Add `BatchNormKind.BATCH_NORM_INFERENCE` in the existing normalization operation package.
- Add one immutable public `BatchNormInferenceAttrs(channelAxis, epsilon)` value containing only
  a normalized non-negative channel axis and exact typed epsilon.
- Give `BATCH_NORM_INFERENCE` exactly one fixed five-input/one-output signature.
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
  allow that axis at any logical position without assuming NCHW, NHWC, or a resolved layout.
- Require scale, bias, running mean, and running variance to be exact rank-one per-channel vectors,
  with local static mismatch rejection and explicit deferred symbolic equality obligations.
- Require all five inputs to be floating, promote them in exact occurrence order, and require
  epsilon to have the exact result type.
- Select the exact inference formula, running-variance interpretation, computation format, empty,
  negative-variance, NaN, infinity, signed-zero, overflow, reassociation, and determinism policies.
- Produce one fresh unlabeled, storage-free, unresolved-layout result with exact input Shape,
  combined gradient eligibility, ordered provenance, and output index zero.
- Update focused Javadocs, Tensor and Compile API references, glossary, capability baseline,
  master plan, roadmap, task evidence, and public-surface tests in the same overall change.

## Out of scope

- training-time batch mean or variance, reduction axes, statistic calculation, training mode,
  evaluation mode, mode flags, momentum, running-stat update formulas, mutation, saved mean,
  saved variance, saved inverse standard deviation, state inputs/outputs, or multi-output training
- any decision about task 0021C's operation kind, attributes, signatures, outputs, public surface,
  formulas, update policy, or implementation
- optional affine operands, no-affine, scale-only, bias-only, implicit ones/zeros, nullable or
  optional operands, parameter initialization, layer/module objects, parameter ownership, or
  trainable registries
- scalar or broadcast affine/statistic operands, per-element or grouped parameters, multiple
  channel axes, inferred channel axis, a batch-axis attribute, NCHW/NHWC-specific aliases, or a
  general options object
- rank-zero or rank-one data, default channel axis, default epsilon, untyped binary64 epsilon,
  epsilon Tensor input, configurable output type, or configurable computation/accumulator type
- rejection by reading running-variance values, a non-negative-value proof object, variance clamp,
  absolute-value repair, standard-deviation input, inverse-variance input, correction conversion,
  or denominator sentinel
- integral, BOOL, FLOAT16, quantized, sparse, complex, or unsigned inputs; implicit or explicit
  cast-producer insertion
- value reads, eager evaluation, host allocation, result storage, resolved layout, input mutation,
  materialization, interning, or canonicalization
- algorithms, fixed traversal/evaluation order, fused multiply-add selection, gradients, adjoints,
  compiler capture, graph-wide constraint solving, binding, decomposition, fusion, backend
  support, lowering, prepare, runtime, execution, tolerances, conformance, or integration
- changes to layer/RMS normalization, softmax, reductions, typed scalar, Shape, signature,
  factory/producer/provenance foundations, architecture, dependencies, Gradle, other modules, or
  detailed specifications for later tasks

## Public and operation contracts

### Semantic kind, attributes, and signature

`BatchNormKind` contains exactly `BATCH_NORM_INFERENCE` in this task. It does not extend or reuse
`LayerNormKind`, `RmsNormKind`, their attributes, or their trailing-Shape semantics. The kind
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

Its compact constructor validates in component order: channel axis, then epsilon. It requires a
non-negative normalized axis; requires non-null floating epsilon; and requires epsilon to be
finite and strictly greater than positive zero in its exact BFLOAT16, FLOAT32, or FLOAT64
representation. It retains the exact immutable epsilon reference. Exact intrinsic failures are:

```text
IllegalArgumentException("channelAxis must be non-negative: <channelAxis>")
NullPointerException("epsilon")
IllegalArgumentException("epsilon must have a floating data type, but was <dataType>")
IllegalArgumentException("epsilon must be finite and positive: <epsilon>")
```

Positive zero, negative zero, negative finite values, either infinity, and every NaN encoding fail
the final epsilon check. The record stores no Tensor, raw axis, input rank, channel extent,
operand-presence flag, result, layout, gradient, compiler, backend, runtime, or training state.

### Public surface and ordered operand roles

The receiver is the data input. The five producer positions have stable roles:

| Position | Role | Required Shape |
|---|---|---|
| 0 | input data | rank at least two, unchanged result Shape |
| 1 | scale | rank-one vector `[C]` |
| 2 | bias | rank-one vector `[C]` |
| 3 | running mean | rank-one vector `[C]` |
| 4 | running variance | rank-one vector `[C]` |

`C` is the input extent at normalized `channelAxis`. Scale and bias are mandatory explicit affine
operands. Running mean and running variance are mandatory explicit estimated statistics. There is
no overload without any one of them, nullable value, `Optional<Tensor>`, hidden constant,
attribute-held affine/statistic value, attrs-taking public overload, static convenience, alias,
or training/evaluation switch.

The method constructs expression metadata, not a layer. It creates, initializes, owns, registers,
or mutates no parameter or statistic and does not infer whether an input is trainable state.

### Rank, channel axis, and layout-neutral Shape contract

Input rank `R` must satisfy `R >= 2`. The caller channel axis is normalized exactly once through
the current `Shape.normalizeAxis` rule, so valid raw values are in `[-R, R - 1]`. The normalized
non-negative value is retained in attributes. Every logical input axis is eligible to be the
channel axis; there is no distinguished batch-axis position and no NCHW, NHWC, NCW, NCL, or
spatial-rank assumption. Every coordinate sharing the same channel coordinate uses the same four
per-channel values.

The result retains the exact input Shape reference and every exact Dimension reference. Input
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

### Parameter/statistic Shapes and symbolic validation

Each of scale, bias, running mean, and running variance must have rank exactly one. Role checks
occur in that order. Exact rank failures are:

```text
batchNormInference scale rank must be one, but was <rank>
batchNormInference bias rank must be one, but was <rank>
batchNormInference runningMean rank must be one, but was <rank>
batchNormInference runningVariance rank must be one, but was <rank>
```

For each role, compare its sole Dimension to the input Dimension at normalized channel axis:

- structurally equal static, dynamic, or expression Dimensions pass;
- unequal static Dimensions fail locally;
- any unequal pair involving an unresolved Dimension is accepted with an equality obligation for
  later compiler capture/binding because the result Shape remains exactly derivable; and
- no parameter/statistic Shape or Dimension is copied, expanded, broadcast, or rewritten.

Static failures, checked in role order, are:

```text
batchNormInference scale channel dimension mismatch: input=<inputDimension>, scale=<scaleDimension>
batchNormInference bias channel dimension mismatch: input=<inputDimension>, bias=<biasDimension>
batchNormInference runningMean channel dimension mismatch: input=<inputDimension>, runningMean=<runningMeanDimension>
batchNormInference runningVariance channel dimension mismatch: input=<inputDimension>, runningVariance=<runningVarianceDimension>
```

The four vectors need not contain the same exact Shape object. Equality to the one input channel
extent is the semantic obligation. Ordinary broadcasting, scalar operands, `[1, C]`, or a
full-rank singleton-expanded Shape are not accepted.

### Empty and channel-zero behavior

A static channel extent zero is valid only with four static `[0]` vectors. The result then has zero
elements and no formula is evaluated. An unresolved channel extent may later bind to zero if every
deferred vector equality also binds to zero. A zero on any non-channel axis also makes the result
empty; the four vectors still describe the channel extent and are validated normally. Empty
results contain no normalized values, denominator, NaN, infinity, or signed-zero result.

There is no division-by-count or batch-statistic reduction in inference, so an empty non-channel
axis creates no special statistic rule. No eager storage inspection is used to discover emptiness
beyond static Shape metadata.

### Formula, epsilon placement, and running variance

For an output coordinate whose channel is `c`, define:

```text
centered     = input - runningMean[c]
denominator  = sqrt(runningVariance[c] + epsilon)
standardized = centered / denominator
output       = standardized * scale[c] + bias[c]
```

Epsilon is added to running variance inside the square root. Affine order is multiply by scale,
then add bias. The supplied running variance is interpreted as an estimated per-channel variance
and used directly. It is not a standard deviation, inverse variance, inverse standard deviation,
sum of squares, or correction-bearing sample statistic. Inference performs no population/sample
correction conversion, recomputation from input, momentum update, or mutation.

For FLOAT64 data `[1, 2]` in one channel with scale `2`, bias `0.5`, running mean `1`, running
variance `4`, and epsilon `1e-5`, the outputs are approximately
`[0.5, 1.4999987500023437]`. This is a mathematical illustration only; construction reads no
values and computes no result.

This explicit five-input, one-output inference formula follows official
[ONNX BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html).
Synaptik deliberately uses an explicit layout-neutral channel axis, exact rank-one vectors, and
exact typed epsilon rather than ONNX's fixed channel position, defaults, or combined mode surface.

### Negative running variance

Model construction does not read or prove running-variance values and therefore accepts a Tensor
whose eventual values may be negative. There is no clamp, absolute value, eager rejection, hidden
validation output, or replacement denominator. The formula is authoritative:

- finite `runningVariance[c] + epsilon < 0` produces a NaN denominator and NaN output;
- an exact zero sum produces a zero denominator, so centered zero yields NaN and non-zero centered
  values yield signed infinity before affine operations;
- a negative variance greater than `-epsilon` leaves a positive radicand and follows the ordinary
  formula; and
- negative infinity produces NaN through the square root.

This policy does not claim that negative variance is a valid trained statistic. It defines
observable formula behavior without adding value-reading validation to model construction.

### Data types, promotion, and computation format

Input, scale, bias, running mean, and running variance must each be BFLOAT16, FLOAT32, or FLOAT64.
Derive result type by applying `DataTypePromotion.promoteFloating` in exact producer occurrence
order: input with scale, then bias, then running mean, then running variance. Promotion changes
result metadata only and inserts no cast producer.

Exact type failures are:

```text
batchNormInference input must have a floating data type, but was <dataType>
batchNormInference scale must have a floating data type, but was <dataType>
batchNormInference bias must have a floating data type, but was <dataType>
batchNormInference runningMean must have a floating data type, but was <dataType>
batchNormInference runningVariance must have a floating data type, but was <dataType>
batchNormInference epsilon data type must match result data type: epsilon=<epsilonType>, result=<resultType>
```

Epsilon must exactly equal the promoted result type. There is no reduction accumulator in
inference. Formula arithmetic treats all five inputs and epsilon in FLOAT32 when the result is
BFLOAT16 or FLOAT32, and in FLOAT64 when the result is FLOAT64; the final value is rounded to the
result format. No narrower computation format or independently configurable accumulator/result
type is allowed.

For finite values, the semantic target is the stated real formula rounded to the result format.
A conforming implementation may use equal-or-wider intermediates, reassociate operations, or use
fused multiply-add only when later conformance tolerances and the special-value classes below are
preserved. Model selects no algorithm. Bitwise equality, one fixed arithmetic grouping, NaN
payload/sign preservation, and cross-backend identical finite rounding are not promised.

### NaN, infinity, signed zero, overflow, and determinism

- A NaN participating at a coordinate in input, scale, bias, running mean, running variance, or
  epsilon-derived arithmetic propagates to NaN output. No later affine zero suppresses it.
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
  signed-zero rules. No canonical zero sign is imposed.
- Finite subtraction, addition, multiplication, or division may overflow or underflow in the
  selected computation format and then follows its ordinary infinity, zero, and NaN rules.
- Empty results contain no value-level special cases.

Because inference is coordinatewise and performs no reduction, there is no traversal-order
dependency. Separately constructed equal requests remain distinct producers. Finite rounding may
still differ under permitted grouping or fused operations, so bitwise cross-backend determinism is
not promised; the special-value classes above are deterministic semantic constraints.

### Result metadata, provenance, and identity effects

The descriptor uses the ordered-promoted result type, exact input Shape reference, unresolved
layout, and `requiresGrad` equal to the logical OR of all five input descriptors' eligibility.
Every result is fresh, unlabeled, and storage-free. Inputs and their descriptors, IDs, labels,
storage, layout, gradient eligibility, and provenance are not mutated.

Each success creates one `Operation`, one `TensorProducer`, one output descriptor, one Tensor
wrapper, and one Tensor ID. Provenance output index is zero and exact ordered inputs are
`[input, scale, bias, runningMean, runningVariance]`. No hidden constant, saved statistic, sibling
descriptor, additional wrapper, state token, or extra ID is created. Future compiler or training
work cannot retroactively change this inference producer's fixed one-output contract.

### Validation and construction order

`TensorBatchNormInferenceExpressions` exposes exactly one package-private entry matching the
public receiver method. It validates in this order:

1. null-check `input`, `scale`, `bias`, `runningMean`, `runningVariance`, then `epsilon`;
2. validate floating eligibility for input, scale, bias, running mean, then running variance;
3. require input rank at least two;
4. normalize the caller channel axis exactly once through `Shape.normalizeAxis`;
5. require rank one for scale, bias, running mean, then running variance;
6. validate static/symbolic channel compatibility for those four roles in the same order;
7. promote the five input data types in exact occurrence order;
8. validate epsilon exact result type;
9. construct `BatchNormInferenceAttrs`, descriptor, operation, and delegate exactly once.

Null failures use exact parameter names. The attributes constructor defensively repeats its
intrinsic normalized-axis and epsilon validation. Existing Shape, promotion, descriptor,
operation, producer, and identifier failures retain their current messages. Every local failure
occurs before factory delegation and consumes no Tensor ID, producer, descriptor wrapper, or
result Tensor.

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
- [Layer normalization](0021-layer-normalization-semantics-and-tensor-expressions.md)
- [RMS normalization](0021a-rms-normalization-semantics-and-tensor-expressions.md)
- [Operation signatures](0018k-operation-signature-and-construction-hardening.md)
- [Shared producer provenance](0018l-shared-multi-output-tensor-provenance.md)
- [Typed scalar values](0018n-typed-scalar-value-contract.md)
- [Multi-axis and statistical reductions](0018v-multi-axis-and-statistical-reductions.md)

## Architecture constraints

- Work stays in model plus its documentation/planning. `Tensor` remains public mutable API state,
  not graph intermediate representation.
- Batch-normalization inference operation types contain backend-independent semantic parameters
  only and do not import Tensor, graph, compiler, training, runtime, prepare, or backend types.
- Package direction is tensor helper to normalization operation/datatype/shape. No reverse edge,
  registry, service, layer owner, parameter owner, statistic owner, or state owner is introduced.
- Explicit running-statistic Tensors are immutable logical inputs to this occurrence. The
  operation does not own or mutate them and has no hidden training/evaluation lifecycle.
- Compiler owns capture, deferred equality proof, binding validation, gradients, saved values,
  adjoints, and legal decomposition. Backend prepare owns algorithms, lowering, specialization,
  fusion, kernels, and tolerance satisfaction. Runtime executes prepared work without consuming
  `Operation` on its hot path.
- No architecture, focused-architecture, dependency, lifecycle, training, Gradle, or cross-module
  change is authorized.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.model.operation.normalization` — add inference-only batch-
  normalization semantics beside unchanged softmax, layer-normalization, and RMS-normalization
  families.
- `io.github.pho001.synaptik.model.datatype`
- `io.github.pho001.synaptik.model.shape`
- `io.github.pho001.synaptik.model.operation`
- `io.github.pho001.synaptik.model.tensor`

Packages added: None.

Type placement:

- `...operation.normalization.BatchNormKind` — public inference semantic identity and fixed
  occurrence signature.
- `...operation.normalization.BatchNormInferenceAttrs` — public normalized channel-axis and exact
  typed epsilon parameters.
- `...tensor.TensorBatchNormInferenceExpressions` — package-private validation, promotion, Shape,
  descriptor, and one-output provenance construction owner.
- `...tensor.Tensor` — established public fluent receiver facade.

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
  full model run discovered its pre-existing exact-count assertion.

Documentation/planning (7):

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged unless inaccurate: Runtime and Training APIs; layer/RMS normalization, softmax,
reductions/statistics, typed scalar, Shape, operation/signature, Tensor/factory/producer/provenance,
and other operation contracts; architecture/ADRs/tests; conformance/integration; Gradle; other
modules.

## Maximum scope

Exactly 20 paths maximum: four production, nine test, and seven documentation/planning paths.
The original task was one path above the planning guide's normal 18-path upper guardrail because the one new
public method requires six established exact public-count assertions to advance together, while
the semantic kind, typed attrs, helper, focused semantics/expression tests, and mandatory seven-
path documentation handoff are inseparable for an actionable public capability. The user
explicitly authorized that cohesive 19-path scope. During the first full model run, the
pre-existing `TensorRmsNormExpressionTest` independently asserted the same public count and failed
only because it still expected 183. The user then explicitly authorized that necessary count-only
update as the sole path-20 addition. Stop for path 21, another production type,
test, document, helper change, architecture/Gradle/cross-module work, optional affine form,
training/state behavior, multi-output behavior, or a later detailed task specification.

## Javadoc and documentation requirements

- Fully document kind, attrs, helper, and Tensor method: five ordered roles, layout-neutral channel
  axis, exact vector Shapes, formula/running-variance/epsilon contract, promotion/computation,
  empty/negative/special values, metadata/provenance/ID effects, validation/failures, and lifecycle
  boundaries.
- Every input, non-void result, and expected failure has complete `@param`, `@return`, and
  `@throws` documentation.
- Tensor API gets the exact signature, operand/Shape table, one finite numerical example, static
  mismatch, symbolic deferral, empty/channel-zero and negative/special-value policies, provenance,
  and current-model versus planned compiler/execution boundary.
- Compile API records batch-normalization inference metadata as current after implementation and
  leaves capture, deferred constraint proof, gradients, saved values, decomposition, lowering,
  training transition, and execution planned.
- Review glossary terms batch normalization, batch-normalization inference, channel axis, running
  mean, running variance, estimated statistic, epsilon, affine transform, computation format,
  producer, and saved statistic; add/refine only reusable distinctions needed by the API.
- Synchronize statuses: 0021 and 0021A remain Complete; 0021B is Ready during implementation and
  becomes Complete only after every criterion passes; 0021C and 0022–0024 remain Draft without
  detailed specifications.
- Record reasoned no-change conclusions for Runtime/Training APIs, related contracts,
  architecture, conformance/integration, Gradle, and other modules.

## Acceptance criteria

- Exact `BATCH_NORM_INFERENCE`, one fixed attrs/signature variant, and one canonical receiver method
  exist; existing normalization families remain unchanged and public Tensor count is 184.
- Exact mandatory ordered roles and one output are locked; no optional/hidden operand, static
  entry, alias, attrs overload, mode flag, saved output, or mutation exists.
- Rank-at-least-two input, one normalized layout-neutral channel axis, exact rank-one parameter/
  statistic Shapes, static mismatch rejection, symbolic equality deferral, exact input Shape
  retention, and empty/channel-zero rules match this task.
- Ordered five-input floating promotion, exact-result-typed finite positive epsilon, and selected
  FLOAT32/FLOAT64 computation formats pass without cast insertion or value evaluation.
- Exact inference formula, direct estimated-variance interpretation, epsilon placement, negative
  variance, NaN/infinity/signed-zero, overflow, reassociation, rounding, and determinism policies
  are explicit and internally consistent.
- Requires-grad OR, unresolved layout, freshness, no label/storage/mutation, exact input order, one
  output/index zero, and exact one-ID/producer/wrapper effects pass.
- Validation order and exact task-owned messages pass; every pre-factory failure consumes no ID.
- No training statistic/update/momentum/multi-output detail, gradient/compiler/algorithm/backend/
  runtime behavior, architecture/dependency/build/cross-module change, or later spec is added.
- Focused tests, the recorded discovery model run and one final stable model suite,
  Javadoc/docs/link/anchor/fence/newline/whitespace, official-link, exact 20-path/package/public-
  surface/status audits, and `git diff --check` pass.
- A separate clean documentation-focused pass reuses Java evidence, finalizes all affected
  Javadocs/docs/glossary impact, and records reasoned no-change conclusions before completion.

## Tests / validation

Focused development validation:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceSemanticsTest --tests io.github.pho001.synaptik.model.tensor.TensorBatchNormInferenceExpressionTest --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest --tests io.github.pho001.synaptik.model.datatype.ScalarValueTest --tests io.github.pho001.synaptik.model.shape.ShapeTest --tests io.github.pho001.synaptik.model.tensor.TensorProducerTest --tests io.github.pho001.synaptik.model.tensor.TensorProvenanceTest
```

After executable Java stabilizes, exactly once:

```bash
./gradlew :modules:model:test
```

Focused tests cover exact kind/attrs/signature, public surface, ordered operand roles, attrs
validation, axis normalization, rank/static/symbolic/empty Shapes, promotion/computation and
formula policies as inspectable semantic contracts, negative/special values without eager
execution, metadata, validation/ID effects, freshness, and exact one-output provenance.

Documentation pass after final Javadoc:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation-focused pass also runs targeted Markdown link/anchor/fence/final-newline and
trailing-whitespace checks, verifies the official ONNX link, audits exact 20-path scope/package
placement, checks public count/signature and synchronized statuses, and records reused Java
evidence. Repository-wide validation is deferred to the selected modern-operation checkpoint
after task 0022 and CI because this task changes one module without dependencies, architecture,
shared build configuration, or executable backend behavior.

## Dependencies

- 0001–0002: DataType, Shape, Dimension, static/symbolic, rank, and axis-normalization contracts.
- 0005–0007 and 0011–0013: operation, descriptor, Tensor, factory, and provenance foundations.
- 0018K: exact attributes-class signatures and occurrence cardinality.
- 0018L: shared producer/output-index provenance, used here in one-output form.
- 0018N: exact typed scalar representation for epsilon.
- 0018V: variance, correction, computation/accumulation, empty, special-value, and determinism
  vocabulary precedents; inference consumes an explicit variance rather than reducing input.
- 0021–0021A: typed epsilon, normalization metadata, one-output provenance, documentation, and
  lifecycle-separation precedents without reusing their trailing-Shape contracts.

All dependencies are Complete.

## Follow-up tasks

- 0021C (Draft) remains the separately bounded training/statistic-transition frontier. This task
  establishes only that any future training work must not mutate or hide state and must not alter
  the completed one-output inference occurrence. No 0021C signature, formula, output, momentum,
  saved-statistic, or public-surface decision is made here.
- 0022–0024 remain Draft at their established IDs. Task 0023 owns any batch-normalization adjoint
  or compiler-generated saved-statistic meaning justified by compiler implementation.

## Architecture impact

Expected impact: None.

The task adds model-owned inference semantics and public expression metadata inside existing
package and dependency boundaries. If implementation requires hidden mode/state, mutation, a
cross-module type, architecture/dependency change, or compiler/backend/runtime behavior, stop and
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

## Documentation-agent handoff

Provide the clean documentation context with this task, the complete implementation diff, exact
focused/final Java commands and results, affected inference semantics/public method, architecture
and 20-path constraints, expected Tensor/Compile API and glossary/planning impact, the official
ONNX link, and required Javadoc/Markdown/scope/status/whitespace evidence. It must inspect the
actual source/tests, not rely only on the handoff summary, and record reasoned no-change
conclusions.

## Local decisions

- Keep `BatchNormInferenceAttrs` limited to normalized channel axis and exact typed epsilon;
  occurrence-specific Tensor roles, rank, Shape, promotion, and result facts remain at the public
  expression boundary.
- Use direct running variance with epsilon inside the square root and exact ordered five-input
  promotion. Negative running values are not read during construction and receive no repair.
- The first full model run exposed one additional pre-existing exact public-count assertion in
  `TensorRmsNormExpressionTest`. The user authorized only its 183-to-184 update as path 20; no
  other scope expansion was made.

## Known limitations

- This task constructs model metadata only. Compiler capture, deferred equality proof, gradients,
  saved values, decomposition, backend preparation, numerical execution, and conformance remain
  unimplemented in their owning layers.
- Training-time batch statistics, running-statistic transition, momentum, and multi-output saved
  statistics remain Draft task 0021C without a detailed specification.

## Validation evidence

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
  accurate and needed no further edit.
- `./gradlew :modules:model:javadoc` passed `BUILD SUCCESSFUL in 1s` after the final Javadoc review.
- The targeted Markdown checker resolved 617 local links, including 166 heading anchors, across
  all seven documentation/planning paths with zero failures. Separate checks found balanced
  backtick and tilde fences, final newlines, no trailing whitespace, and zero failures. Manual
  recalculation confirmed the FLOAT64 `[1, 2]` example results `0.5` and approximately
  `1.4999987500023437`.
- `curl -I -L --max-time 20 https://onnx.ai/onnx/operators/onnx__BatchNormalization.html`
  returned HTTP 200 for the official ONNX BatchNormalization reference.
- `javap -public` inspection found exactly 184 public `Tensor` methods and exactly one
  `batchNormInference(int, Tensor, Tensor, Tensor, Tensor, ScalarValue)` receiver returning
  `Tensor`. Source/test inspection confirmed the sole `BATCH_NORM_INFERENCE` enum value, exact
  fixed five-input/one-output signature, one package-private helper entry, ordered provenance,
  and no static public entry, alias, optional operand, training flag, or auxiliary output.
- Final inventory found exactly 20 authorized paths: four production, nine tests, and seven
  documentation/planning paths. Production types remain in the planned normalization/tensor
  packages with model-only imports; no architecture, dependency, Gradle, cross-module, later-task
  specification, training, compiler, backend, runtime, conformance, or integration path changed.
- Runtime API remains accurate unchanged because no prepared execution, storage, state, or run
  behavior was added. Training API remains accurate unchanged because no parameter, optimizer,
  statistic transition, momentum, saved output, or training mode was added. Related operation,
  layer/RMS normalization, scalar, Shape, signature, factory, producer, and provenance contracts
  remain accurate unchanged because the new family composes their established boundaries without
  modifying them. Architecture/ADRs/tests, conformance/integration, Gradle, and other modules need
  no change because ownership, dependencies, build behavior, backend behavior, and end-to-end
  execution did not change.
- Final status audit confirmed tasks 0021, 0021A, and 0021B Complete; task 0021C and tasks
  0022–0024 Draft without detailed specifications. Final fence/newline/whitespace and
  `git diff --check` validation passed.

## Implementation notes

- Added `BatchNormKind`, `BatchNormInferenceAttrs`, one package-private construction helper, and
  one canonical public receiver. The helper validates all five floating operands, rank and
  channel-vector relationships, ordered promotion, exact epsilon, result metadata, freshness,
  and one-output provenance before its sole factory delegation.
- Finalized Tensor and Compile API references, glossary terminology, capability baseline, task,
  master plan, and roadmap. Current model behavior is separated explicitly from planned compiler,
  training, backend, runtime, and execution work.

## Completion summary

- Completed changes: stateless explicit five-input batch-normalization inference semantics,
  attributes, public expression construction, tests, Javadocs, and synchronized documentation.
- Files changed or created: exactly 20 authorized paths — four production, nine tests, and seven
  documentation/planning files listed above.
- Tests and validation: focused suite, authorized isolated count correction, final 919-test model
  suite, model Javadoc, 617-link/166-anchor Markdown, official ONNX link, public surface, exact
  scope/package/status, fence/newline/trailing-whitespace, and `git diff --check` all passed.
- Documentation-agent review: completed in clean context
  `/root/task_0021b_implementation/docs_finalize` using the required profiles and reused Java
  evidence without rerunning successful executable suites.
- Documentation impact: Tensor API, Compile API, glossary, capabilities, task, master plan, and
  roadmap finalized; Runtime and Training APIs and related architecture/docs remain accurate
  unchanged for the reasons recorded above.
- Javadoc review: all four production contracts are complete and consistent with source/tests;
  generated model Javadoc passed without requiring another Java-source edit.
- Glossary impact: added the reusable batch-normalization inference, channel-axis, running-
  statistic, direct-variance, computation-format, epsilon, affine, and saved-statistic
  distinctions.
- Unresolved issues: None.
- Follow-up required: None for this task; Draft task 0021C remains separately planned work.

Status: Complete
