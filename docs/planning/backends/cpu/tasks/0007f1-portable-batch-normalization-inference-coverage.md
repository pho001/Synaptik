# Task 0007F1: Portable [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[Normalization](../../../../glossary.md#term-normalization) [Inference](../../../../glossary.md#term-inference) Coverage

## Reader context

This task records stateless BATCH_NORM_INFERENCE: five ordered inputs supply data, affine vectors,
and running statistics, and one output receives direct coordinatewise results. Model 0021B owns the
formula and descriptors; Compiler validates the first-class occurrence. `CpuBatchNormInferenceIr`,
`CpuBatchNormInferenceLowering`, and `CpuBatchNormInferenceEmitter` keep arbitrary channel-axis
geometry separate from trailing Layer/RMS reductions and from training transitions.

Here, preparation chooses channel-owned or non-channel-owned ranges before artifact realization.
Generated code hoists channel reads and square roots; it neither updates running inputs nor
allocates statistic storage. Cold binding validates storage before direct writes, without rollback.
Clean Java is the structural algorithm/overhead oracle, while independent accuracy and special-value
checks are the semantic oracle.

Current-topic navigation (not a replacement for the historical record):

- [current stateless BatchNorm inference inputs](../../../../reference/tensor/normalization.md#batch-normalization-inference-expressions) — Names data, scale, bias, and read-only running statistics at the Tensor expression boundary.
- [current CPU preparation and invocation handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) — Names the current analysis, shared assignment, finalization, cold binding, and Runtime participants.

## Status

Complete

## Goal

Execute exactly one fully static, resolved-[layout](../../../../glossary.md#layout), first-class `BATCH_NORM_INFERENCE` [occurrence](../../../../glossary.md#term-occurrence)
on the portable [CPU](../../../../glossary.md#term-cpu) route. Preserve [Model](../../../../glossary.md#term-model) 0021B's stateless five-input/one-output semantics and
[Compiler](../../../../glossary.md#term-compiler) 0005B's validated first-class form through one direct generated coordinatewise body with
arbitrary normalized [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis), exact ordered floating promotion, exact typed [epsilon](../../../../glossary.md#epsilon), explicit
affine and running-[statistic](../../../../glossary.md#term-statistic) reads, channel-hoisted square roots, deterministic [scalar](../../../../glossary.md#term-scalar) or
parallel-scalar ranges, and zero workspace.

## [Mental model](../../../../glossary.md#term-mental-model)

```text
one explicit BATCH_NORM_INFERENCE node
  -> CPU validates one static input Shape and four exact [C] vectors
  -> CPU selects one channel-owned or non-channel-owned range form
  -> each invocation hoists channel values and one square root per channel it covers
  -> each coordinate executes the exact typed inference formula
  -> one direct generated store publishes the result
```

[Scalar](../../../../glossary.md#term-scalar) execution owns channels and computes each [channel](../../../../glossary.md#term-channel) denominator exactly once. Parallel
execution partitions either channels or the flattened non-channel domain, whichever exposes the
larger bounded range domain. A non-channel range repeats each channel denominator at most once per
submitted range, not once per prefix or [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). Both forms keep the channel-[axis](../../../../glossary.md#term-axis) contract
[layout](../../../../glossary.md#layout)-neutral, require no [cache](../../../../glossary.md#term-cache) or workspace, retain channel values in primitive locals across
stores so Java array-[alias](../../../../glossary.md#aliasing-and-arena) conservatism cannot force reloads, and give invocations disjoint output
regions.

## [Scope](../../../../glossary.md#term-scope)

- Recognize exactly one compiled `BatchNormKind.BATCH_NORM_INFERENCE` [occurrence](../../../../glossary.md#term-occurrence) with
  `BatchNormInferenceAttrs` and ordered
  `[input, scale, bias, runningMean, runningVariance] -> [output]`.
- Preserve the first-class semantic identity. Admit no training occurrence and never infer,
  reconstruct, fuse, or relabel batch normalization from decomposed arithmetic or reduction
  graphs.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [graphs glossary](../../../../glossary.md#graph))</span>
- Accept BFLOAT16, FLOAT32, and FLOAT64 independently at all five input positions. Promote in
  exact occurrence order: input with scale, then bias, then running mean, then running variance.
  Require output and epsilon to use the resulting type and require output gradient eligibility to
  equal the logical OR of all five input descriptors.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [bias glossary](../../../../glossary.md#term-bias); [mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [epsilon glossary](../../../../glossary.md#epsilon); [gradient glossary](../../../../glossary.md#term-gradient); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Require fully static resolved descriptors, input rank at least two, a normalized non-negative
  channel axis within rank, four exact rank-one vectors whose extent equals the selected channel
  extent, exact input/output Shape equality, non-negative checked layouts, and an injective
  writable output.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [vectors glossary](../../../../glossary.md#term-vector); [extent glossary](../../../../glossary.md#term-extent); [Shape glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [injective glossary](../../../../glossary.md#term-injective))</span>
- Derive checked geometry:

  ```text
  prefixCount = product(input extents before channelAxis)
  channelCount = input extent at channelAxis
  suffixCount = product(input extents after channelAxis)
  nonChannelCount = prefixCount * suffixCount
  outputCount = channelCount * nonChannelCount
  ```

  All products use checked `long` arithmetic before resource declaration. A flattened
  non-channel coordinate maps canonically to prefix-major, then suffix-major coordinates.
- Select exactly one range form during [CPU](../../../../glossary.md#term-cpu) preparation, before [artifact](../../../../glossary.md#term-artifact) realization:

  ```text
  scalar:
    CHANNEL_RANGE

  parallel:
    rangeForm = CHANNEL_RANGE when channelCount >= nonChannelCount,
                otherwise NON_CHANNEL_RANGE
    rangeItemCount = rangeForm == CHANNEL_RANGE ? channelCount : nonChannelCount
    coordinatesPerRangeItem = rangeForm == CHANNEL_RANGE
        ? nonChannelCount
        : channelCount
    minimumRangeItemsPerWorker = max(
        1,
        ceil(existingMinimumElementsPerWorker / coordinatesPerRangeItem))
    desiredRanges = min(usableParallelism,
                        ceil(rangeItemCount / minimumRangeItemsPerWorker))
    selectedRangeCount = max(1, desiredRanges)
  ```

  Empty output retains the existing no-range path. `CHANNEL_RANGE` interprets `start/end` as a
  half-open channel interval and traverses the complete non-channel domain for each owned channel.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval))</span>
  `NON_CHANNEL_RANGE` interprets them as a half-open flattened non-channel interval and traverses
  every channel for that owned interval. The range form is a CPU-private code-shaping
  specialization/cache fact. It is not a runtime choice, measured candidate, tuning decision,
  fixed-Shape variant, or additional generated artifact for one preparation. Range orchestration
  uses <span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache); [runtime glossary](../../../../glossary.md#term-runtime))</span>`minimumRangeItemsPerWorker`, so the existing minimum-work threshold remains measured in
  output coordinates rather than accidentally treating one channel or one non-channel coordinate
  as equal work.
- Treat channel extent zero or any zero non-channel extent as an empty output. Retain truthful
  input/output declarations, but perform no input read, channel-vector read, output write,
  generated invocation, square root, or worker submission.<span data-glossary-navigation="true"> ([square root glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>
- In `CHANNEL_RANGE`, for each owned [channel](../../../../glossary.md#term-channel) read and promote `scale[c]`, `bias[c]`,
  `runningMean[c]`, and `runningVariance[c]` once, compute the denominator once, then traverse all
  non-channel coordinates in canonical prefix-major/suffix-major order. Scalar execution
  therefore performs exactly <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Scalar glossary](../../../../glossary.md#term-scalar))</span>`channelCount` square roots.
- In `NON_CHANNEL_RANGE`, one invocation traverses channels in increasing order. For each channel
  it performs the four promoted channel reads and denominator calculation once, then traverses
  only its owned non-channel interval. Parallel execution therefore performs at most
  `channelCount * submittedRangeCount` square roots. It never performs one square root per prefix,
  suffix position, or output coordinate. The bounded repetition preserves useful parallelism for
  small channel counts while partitioning the larger non-channel domain for locality.
- Preserve the exact [inference](../../../../glossary.md#term-inference) formula and [operation](../../../../glossary.md#operation) order for each [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix):

  ```text
  centered     = input - runningMean[c]
  radicand     = runningVariance[c] + epsilon
  denominator  = sqrt(radicand)
  standardized = centered / denominator
  scaled       = standardized * scale[c]
  output       = scaled + bias[c]
  ```

  Epsilon is inside the square root. Running variance is consumed directly as an estimated
  variance. There is no correction conversion, clamp, absolute-value repair, reciprocal cache,
  affine rewrite, fused multiply-add, or <span data-glossary-navigation="true"> ([fused multiply-add glossary](../../../../glossary.md#term-fused-multiply-add-fma))</span>`input * alpha + beta` [reassociation](../../../../glossary.md#term-reassociation).
- Treat all five reads and epsilon in FLOAT32 when the result is BFLOAT16 or FLOAT32, and in
  FLOAT64 when the result is FLOAT64. Round each subtraction, addition, square-root result,
  division, multiplication, and final addition at that computation-format boundary. Encode
  BFLOAT16 once at the final store; FLOAT32 and FLOAT64 use their matching primitive store.
- Preserve ordinary floating behavior from Model 0021B. Negative radicands and negative-infinite
  running variance produce NaN through square root; exact zero radicands permit zero denominators;
  positive-infinite variance produces a positive-infinite denominator; input/statistic/affine NaN,
  infinities, signed zeros, overflow, and underflow follow the stated operation order. No later
  affine zero may suppress an already produced NaN.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [NaN glossary](../../../../glossary.md#term-nan); [statistic glossary](../../../../glossary.md#term-statistic); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow))</span>
- Deduplicate exact repeated logical input [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values in first-occurrence order and retain an
  immutable semantic-position-to-boundary map. Declare each unique input once and exactly one
  output. Input/input alias is read sharing; output/input overlap is always rejected before any
  mutation or worker submission.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- After successful cold validation, each generated store publishes its coordinate directly to the
  ordinary output carrier. There is no shadow output, commit phase, or rollback. Model-defined
  non-finite arithmetic does not throw; an unrelated later generated/worker failure follows the
  existing execution contract and may expose already written coordinates.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [commit glossary](../../../../glossary.md#term-commit); [rollback glossary](../../../../glossary.md#term-rollback); [finite glossary](../../../../glossary.md#term-finite))</span>
- Support each unique boundary's exact typed [heap](../../../../glossary.md#term-heap) array or native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment) [carrier](../../../../glossary.md#term-storage-carrier):
  `short[]`, `float[]`, or `double[]` according to that boundary's own type. Support all
  heap/segment combinations, arbitrary legal offsets, positive or zero input strides, and an
  arbitrary injective output layout. Cold binding resolves concrete carriers once; generated hot
  work performs no storage discovery or carrier switch.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [binding glossary](../../../../glossary.md#term-binding))</span>
- Add one focused batch-inference IR, one focused arbitrary-axis lowerer, one direct emitter, and
  one independent reference owner. Reuse existing access-plan, carrier, route, preparation,
  finalization, executable, worker, artifact, and overlap seams without extending the trailing
  Layer/RMS IR or creating a generic normalization interpreter.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [owner glossary](../../../../glossary.md#term-owner); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Select scalar or parallel-scalar execution only. Use zero workspace, zero materialization, zero
  partial/combine state, and zero saved-statistic storage. Worker count and ranges remain cold
  invocation facts; generated code creates and submits no workers.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Emit one deterministic final, field-free, constructor-free generated class per specialization
  with one public static typed `invoke` entry. Its exact parameters are the unique input carriers
  in first-occurrence order, the output carrier, one packed primitive `long[]` geometry, and
  primitive `long start, long end`. It has no scratch [parameter](../../../../glossary.md#term-parameter).
- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly once from 48 to 49. The new family,
  channel-axis geometry, five-position boundary map, epsilon/formula identity, and generated body
  change artifact meaning. Schema 48 and earlier envelopes are safe incompatible misses; retained
  historical evidence remains immutable.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema))</span>
- Use a frozen optimal clean Java implementation with the same selected range form, validation,
  carrier and layout addressing, channel-hoisted reads/root, per-coordinate arithmetic
  boundaries, and stores as the design, review, and performance oracle. The oracle must be
  independently defensible: it may not repeat channel-only work per prefix or coordinate merely
  to mirror generated code. Generated code may call only required JDK primitive,
  raw-bit, <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`Math.sqrt`, and typed `MemorySegment` operations and must call no Synaptik numerical,
  validation, or reference helper.

## Out of [scope](../../../../glossary.md#term-scope)

- `BATCH_NORM_TRAINING`, batch mean or variance reductions, biased or unbiased statistic
  calculation, momentum, next running statistics, saved mean, saved inverse standard deviation,
  statistic transitions, five-output binding, or any CPU 0007F2 work.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [statistic glossary](../../../../glossary.md#term-statistic); [momentum glossary](../../../../glossary.md#term-momentum); [standard deviation glossary](../../../../glossary.md#term-standard-deviation); [binding glossary](../../../../glossary.md#term-binding); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Layer Norm, RMS Norm, softmax, group/instance/local-response/weight/spectral normalization,
  training/evaluation flags, module/parameter ownership, state mutation, backward execution, or
  gradient-policy changes.<span data-glossary-navigation="true"> ([Norm glossary](../../../../glossary.md#term-norm); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [weight glossary](../../../../glossary.md#term-weight); [normalization glossary](../../../../glossary.md#term-normalization); [evaluation glossary](../../../../glossary.md#term-evaluation); [module glossary](../../../../glossary.md#term-module); [parameter glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Optional affine/statistic operands, implicit ones or zeros, scalar/broadcast/full-rank channel
  operands, multiple/inferred/default channel axes, NCHW/NHWC aliases, default or Tensor epsilon,
  or configurable output/computation type.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [channel glossary](../../../../glossary.md#term-channel); [axes glossary](../../../../glossary.md#term-axis); [NCHW glossary](../../../../glossary.md#term-nchw); [NHWC glossary](../../../../glossary.md#term-nhwc); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [Tensor glossary](../../../../glossary.md#tensor); [epsilon glossary](../../../../glossary.md#epsilon))</span>
- Value-based rejection of negative running variance, variance repair, standard-deviation or
  inverse-standard-deviation input, denominator sentinel, relaxed math, or a cross-backend
  bitwise-finite-result promise.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [finite glossary](../../../../glossary.md#term-finite))</span>
- Decomposition, recognition, or fusion of equivalent graphs; fusion with adjacent work; Vector
  API or native/vendor routes; runtime kind/form dispatch; or a scalar-reference Runtime fallback.<span data-glossary-navigation="true"> ([Decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [graphs glossary](../../../../glossary.md#graph); [Vector API glossary](../../../../glossary.md#term-vector-api); [runtime glossary](../../../../glossary.md#term-runtime); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Dynamic or symbolic Shapes, unresolved layouts, runtime channel axis, rank-zero/rank-one input,
  FLOAT16, integral/BOOL/quantized/complex data, negative resolved strides, non-injective output,
  or any checked count/span/address that cannot be represented.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>
- Per-element task submission, per-coordinate square roots or channel-vector reloads, workspace,
  materialized channel vectors, precomputed channel-factor arrays, saved denominators, new
  resource kinds, or hidden caches.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [vector glossary](../../../../glossary.md#term-vector))</span>
- Generated calls to CPU validation/reference code; allocation, boxing, reflection, method
  handles, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`invokedynamic`, dynamic [constants](../../../../glossary.md#term-constant), bootstrap methods, `Object` descriptors,
  collection/map/string lookup, or avoidable semantic dispatch in generated hot work.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Public/shared API, Model, Compiler, Training, Runtime, Prepare, backend-contract, dependency,
  module-boundary, Gradle/toolchain, architecture, conformance, integration, trace, tuning, or
  persistence-format changes beyond the CPU-private generator schema.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [conformance glossary](../../../../glossary.md#term-conformance); [trace glossary](../../../../glossary.md#trace); [persistence glossary](../../../../glossary.md#term-persistence); [schema glossary](../../../../glossary.md#term-schema))</span>
- Reopening historical CPU 0007A1D or creating a detailed specification for CPU 0007F2 or later
  work.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007F Layer/RMS normalization`](0007f-portable-layer-and-rms-normalization-coverage.md)
- [`Model 0021B batch-normalization inference`](../../../modules/model/tasks/0021b-batch-normalization-inference.md)
- [`Model 0021C batch-normalization training`](../../../modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md)
- [`Compiler 0005B normalization inference and gradients`](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [`Prepare 0001 resource declaration`](../../../modules/prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md)
- [`Prepare 0002 finalization handoff`](../../../modules/prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Model owns the first-class kind, five ordered roles, channel-axis meaning, vector Shapes,
  formula, direct running-variance interpretation, epsilon, ordered promotion, computation format,
  result metadata, and exceptional-value classes. Compiler owns capture, descriptor/constraint
  revalidation, and gradients. CPU owns only truthful forward admission, static lowering,
  concrete arithmetic realization, resource/access validation, and generated execution.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [vector glossary](../../../../glossary.md#term-vector); [Shapes glossary](../../../../glossary.md#shape); [variance glossary](../../../../glossary.md#term-variance); [epsilon glossary](../../../../glossary.md#epsilon); [metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [gradients glossary](../../../../glossary.md#term-gradient); [admission glossary](../../../../glossary.md#term-admission); [lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation))</span>
- [CPU](../../../../glossary.md#term-cpu) analysis must select only explicit `BATCH_NORM_INFERENCE`. Neither CPU nor Compiler may
  infer a first-class normalization kind from an equivalent decomposed graph.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [graph glossary](../../../../glossary.md#graph))</span>
- The completed Model and Compiler contracts are implementation inputs, not files to revise. Any
  disagreement in signature, descriptor, channel geometry, promotion, epsilon, or formula is a
  stop condition.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- CPU analysis/lowering chooses the route and computes every static eligibility and exact resource
  fact before shared slot assignment. CPU finalization constructs or reuses the generated artifact
  only afterward. Cold binding validates carriers, thread access, spans, output injectivity, and
  overlap before mutation or worker submission. Runtime receives only a prepared executable.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [binding glossary](../../../../glossary.md#term-binding); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [Runtime glossary](../../../../glossary.md#term-runtime); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- Planning continues to choose only CPU ownership. Shared Prepare remains CPU-blind and assigns
  declared buffer slots without interpreting channel axes, arithmetic, routes, or specialization.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Prepare glossary](../../../../glossary.md#prepare); [buffer glossary](../../../../glossary.md#term-buffer); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Generated code must preserve the frozen optimal clean Java algorithm, selected range/hot-loop/dataflow
  shape, arithmetic boundaries, and avoidable-overhead profile. A reference call, generic bridge,
  per-element callback, packed runtime kind switch, pointwise decomposition, or formula rewrite is
  not a specialized implementation.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [element glossary](../../../../glossary.md#term-element); [callback glossary](../../../../glossary.md#term-callback); [pointwise glossary](../../../../glossary.md#term-pointwise); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Any required public/shared/build/architecture/conformance/integration change, new resource kind,
  workspace, materialization, unresolved semantic decision, or training/statistic-transition work
  is a stop/replan condition.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [materialization glossary](../../../../glossary.md#materialization); [statistic glossary](../../../../glossary.md#term-statistic))</span>

## Current-code evidence

- `BatchNormKind` fixes [inference](../../../../glossary.md#term-inference) to `BatchNormInferenceAttrs`, five inputs, and one output;
  `TensorBatchNormInferenceExpressions` fixes ordered roles, rank/vector Shape rules, promotion,
  exact epsilon type, result Shape, gradient eligibility, and provenance.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [vector glossary](../../../../glossary.md#term-vector); [Shape glossary](../../../../glossary.md#shape); [epsilon glossary](../../../../glossary.md#epsilon); [gradient glossary](../../../../glossary.md#term-gradient); [provenance glossary](../../../../glossary.md#provenance))</span>
- `ReductionNormalizationInference` revalidates the normalized axis, four channel-vector
  equalities, ordered promotion, exact result descriptor, and epsilon. It leaves the occurrence
  first-class. <span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [channel glossary](../../../../glossary.md#term-channel); [descriptor glossary](../../../../glossary.md#term-descriptor); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`NormalizationGradientRules` consumes that exact occurrence for gradients but does
  not define CPU forward execution.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- `CpuCapabilityProvider` currently admits [Softmax](../../../../glossary.md#softmax--log-softmax) and trailing Layer/[RMS normalization](../../../../glossary.md#rms-normalization) only;
  `BATCH_NORM_INFERENCE` therefore fails closed today.
- `CpuPortableKernelIr`, `CpuPartitionLowering`, `CpuPortableRoutePlan`,
  `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`, `CpuPartitionFinalizer`, and
  `CpuPreparedExecutable` enumerate focused family IR/geometry variants and already provide the
  needed one-unit analysis, exact declaration, cold carrier binding, range orchestration, overlap,
  finalization, and direct invocation seams.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [carrier glossary](../../../../glossary.md#term-storage-carrier); [binding glossary](../../../../glossary.md#term-binding))</span>
- `CpuTrailingNormalizationIr` and `CpuTrailingNormalizationLowering` prove ordered mixed-type
  carriers, first-occurrence input deduplication, typed epsilon identity, static non-negative
  layout handling, output injectivity, and optional-family resource accounting. Batch inference
  needs a separate arbitrary-axis zero-workspace geometry rather than extending their trailing
  Shape and Layer-only scratch contract.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [layout glossary](../../../../glossary.md#layout); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [axis glossary](../../../../glossary.md#term-axis))</span>
- `CpuClassFileKernelGenerator`, `CpuKernelSpecialization`, and `CpuGeneratorSchema` currently emit
  one typed direct `invoke` and identify schema 48. The family dispatch and public-entry allowlist
  do not yet include batch inference.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Current generated evidence tests lock deterministic bytes, exact members, direct bodies,
  carrier combinations, schema compatibility, and five-fork generated/direct conventions. CPU
  0007F is the closest mixed-floating normalization precedent; no existing numerical emitter is a
  valid batch-inference implementation.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Package impact

Existing [CPU](../../../../glossary.md#term-cpu)-private packages changed:

- `io.github.pho001.synaptik.backend.cpu` — exact truthful capability [admission](../../../../glossary.md#term-admission).
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — [immutable](../../../../glossary.md#term-immutable) [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[inference](../../../../glossary.md#term-inference) [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — arbitrary-axis channel/non-channel geometry and
  exact unique-boundary derivation.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [channel glossary](../../../../glossary.md#term-channel))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct typed formula body.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent differential [oracle](../../../../glossary.md#term-test-oracle).
- Existing `route.portable`, [`prepare`](../../../../glossary.md#prepare), `executable`, and [`cache`](../../../../glossary.md#term-cache) packages — realization,
  declaration, cold validation/invocation, and schema-49 compatibility.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [schema glossary](../../../../glossary.md#term-schema))</span>

Packages added: None.

Type placement:

- Add `CpuBatchNormInferenceIr` in `internal.ir` for exact kind/form, five semantic input types,
  promoted result, epsilon raw bits, input rank, channel axis, algorithm/arithmetic version,
  semantic-position map, and access patterns.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [rank glossary](../../../../glossary.md#term-rank))</span>
- Add `CpuBatchNormInferenceLowering` in `internal.lowering` for exact static Shape/layout,
  prefix/channel/suffix/non-channel/output counts, canonical coordinate/address mapping,
  first-occurrence unique boundaries, checked spans, the selected range form, and zero-resource
  geometry.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [canonical glossary](../../../../glossary.md#term-canonical-form); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Add `CpuBatchNormInferenceEmitter` in `internal.codegen.emit` for the sole direct channel-hoisted
  and per-coordinate numerical body selected at generation time.
- Add `CpuBatchNormInferenceReferenceKernel` in `internal.reference`; it independently derives
  logical channel coordinates and formula expectations without production lowering, emission,
  binding, or generated helpers.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding))</span>
- Add no validator type because every represented running-variance and affine value has defined
  Model behavior and existing cold structural/overlap validation is sufficient. Add no facade,
  manager, registry, generic normalization interpreter, or change to Layer/RMS emitters.<span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance); [Model glossary](../../../../glossary.md#term-model); [facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- `CpuCapabilityProvider.java` and [CPU](../../../../glossary.md#term-cpu) package Javadoc;
- `internal/ir/CpuPortableKernelIr.java`, new `CpuBatchNormInferenceIr.java`, and IR package
  Javadoc;<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- `internal/lowering/CpuPartitionLowering.java`, new `CpuBatchNormInferenceLowering.java`, and
  lowering package Javadoc;<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new
  `CpuBatchNormInferenceEmitter.java`, existing carrier/loop emission seams only where required,
  and emitter package Javadoc;<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `internal/cache/CpuGeneratorSchema.java`, `CpuKernelSpecialization.java` only if exact entry or
  compatibility validation requires it, and cache package Javadoc;<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [cache glossary](../../../../glossary.md#term-cache))</span>
- `internal/route/portable/CpuPortableRoutePlan.java` and package Javadoc;
- `internal/prepare/CpuPartitionPreparationPlan.java`, `CpuPartitionPreparer.java`,
  `CpuPartitionFinalizer.java`, and package Javadoc;
- `internal/executable/CpuPreparedExecutable.java` and executable package Javadoc; and
- new `internal/reference/CpuBatchNormInferenceReferenceKernel.java`, the minimal existing
  differential seam if required, and reference package Javadoc.

Expected test paths:

- capability and internal-package inventory tests;
- new `CpuBatchNormInferenceIrTest`, `CpuBatchNormInferenceLoweringTest`,
  `CpuBatchNormInferenceGeneratedKernelTest`, and `CpuBatchNormInferenceReferenceTest`;
- generated-class shape/member, specialization/cache, route, prepare/finalization, executable,
  and reference-differential tests only where their exact inventories or behavior change; and<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization); [prepare glossary](../../../../glossary.md#prepare))</span>
- the historical pointwise-ledger test only for its current-schema assertion; retained historical
  evidence resources must not change.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [schema glossary](../../../../glossary.md#term-schema))</span>

Expected documentation/[planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  global roadmap. Architecture, public Tensor/Compile/Training APIs, Model/Compiler plans, shared
  Prepare, backend-contract, conformance, integration, and build files are review-only unless a
  stop condition is reached.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

The complete implementation, tests, and documentation may modify or create at most 46 repository
paths: at most 27 production/[Javadoc](../../../../glossary.md#term-javadoc) paths, 14 test paths, and exactly the five named
documentation/[planning](../../../../glossary.md#term-planning) paths. Exactly four new [CPU](../../../../glossary.md#term-cpu)-private production types are permitted: the
[IR](../../../../glossary.md#term-intermediate-representation-ir), lowerer, emitter, and independent reference [owner](../../../../glossary.md#term-owner) listed above.

This exceeds the planning guide's ordinary file-count guardrail because one truthful CPU family
must update existing [sealed](../../../../glossary.md#term-sealed) IR, [lowering](../../../../glossary.md#lowering), generator, [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache), route, [prepare](../../../../glossary.md#prepare), executable,
package-Javadoc, inventory, and differential seams atomically. The behavioral core remains four
focused types in one [module](../../../../glossary.md#term-module). Splitting those integrations would temporarily advertise, prepare,
or persist a family that cannot execute and would duplicate the final module/evidence gates.

A 47th path, fifth production type, second emitter/validator, generic [normalization](../../../../glossary.md#term-normalization) owner,
Layer/RMS semantic change, public/shared/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration path,
workspace, [materialization](../../../../glossary.md#materialization), or CPU 0007F2 behavior is a stop/replan condition. An omitted expected
path does not authorize an unrelated path.

## Detailed generated-code design boundary

- `CpuBatchNormInferenceIr` is instruction-free and exposes one deterministic family identity.
  Identity includes exact semantic input order/types, result type, epsilon raw bits, input rank,
  normalized channel axis, algorithm/arithmetic version, semantic-position map, and boundary
  access plans. Concrete extents, strides, offsets, carrier instances, slots, ranges, workers,
  graph/run identities, and Tensor values remain cold.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [epsilon glossary](../../../../glossary.md#epsilon); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [rank glossary](../../../../glossary.md#term-rank); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [strides glossary](../../../../glossary.md#term-stride); [carrier glossary](../../../../glossary.md#term-storage-carrier); [graph glossary](../../../../glossary.md#graph); [run glossary](../../../../glossary.md#term-run); [Tensor glossary](../../../../glossary.md#tensor))</span>
- `CpuBatchNormInferenceLowering.Geometry` retains resolved layouts for unique inputs and output,
  prefix/channel/suffix/non-channel/output counts, channel-axis mapping, and packs only primitive
  bases and layout geometry. CPU preparation adds the selected <span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu))</span>`CHANNEL_RANGE` or
  `NON_CHANNEL_RANGE` form. Compatible concrete extents and strides do not enter class identity
  because emitted code reads them from geometry; no fixed-Shape specialization is authorized.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- In either generated form, the channel loop loads the four channel values and computes the typed
  denominator before its coordinate loop. The coordinate loop advances primitive input/output
  coordinates and addresses directly, loads one input, performs the six exact typed arithmetic
  steps, and stores one output. General layouts use typed primitive odometers; proved dense
  heap-array forms may use cold-proved <span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [heap glossary](../../../../glossary.md#term-heap))</span>`int` indexes. No hot operation, carrier, layout,
  channel-axis, or range-form switch remains.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- The emitter bakes exact epsilon bits, result computation format, channel axis, semantic boundary
  map, access form, and selected range form. It does not bake concrete carrier bases, extents,
  strides, offsets, start/end, workers, or values.
- One specialization has exactly one generated artifact. The existing four-candidate/one-artifact/
  zero-shape-variant/zero-unroll-variant budget remains unchanged. Batch inference is never
  materialized and is never Vector-selected in this task.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [inference glossary](../../../../glossary.md#term-inference); [Vector glossary](../../../../glossary.md#term-vector))</span>
- Schema 49 is current-only. A schema-48 or earlier persistent envelope safely misses and may be
  regenerated; no migration reader, compatibility alias, or mutation of historical evidence is
  added.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>

## Acceptance criteria

- Capability admits exactly `BATCH_NORM_INFERENCE` with `BatchNormInferenceAttrs`, five ordered
  inputs, one output, all current BFLOAT16/FLOAT32/FLOAT64 promotion combinations, exact epsilon,
  rank/vector Shape rules, static resolved layouts, result descriptor, and injective output.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon); [rank glossary](../../../../glossary.md#term-rank); [descriptor glossary](../../../../glossary.md#term-descriptor); [injective glossary](../../../../glossary.md#term-injective))</span>
  `BATCH_NORM_TRAINING` and every excluded [occurrence](../../../../glossary.md#term-occurrence) [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).
- Source-backed tests prove an equivalent decomposed graph remains ordinary operations and never
  becomes <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>`CpuBatchNormInferenceIr`; Layer/RMS behavior and capability remain unchanged.
- Lowering records exact five-position types, result type, epsilon bits, rank/channel axis,
  prefix/channel/suffix/non-channel/output counts, first-occurrence unique-boundary map, access
  facts, and zero-resource identity. Preparation records the exact deterministic range form and
  bounded range count. Checked count/span/address arithmetic precedes declarations.<span data-glossary-navigation="true"> ([Lowering glossary](../../../../glossary.md#lowering); [channel glossary](../../../../glossary.md#term-channel); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Semantic/differential coverage includes channel axis first, middle, and last; rank two and
  higher; channel extent one, larger, and zero; zero extents before and after the channel;
  same-type and mixed-floating inputs; dense/general/zero-stride reads; non-contiguous injective
  output; offsets; repeated vector <span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [stride glossary](../../../../glossary.md#term-stride); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) inputs; and representative all-heap, all-segment, and
  mixed-carrier patterns.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Formula tests cover direct running variance, epsilon inside square root, multiply-then-add
  affine order, and the exact computation-format rounding sequence. They include positive and
  negative finite variance, radicand negative/zero/positive, signed zeros, subnormals, raw NaNs,
  both infinities, overflow/underflow, and NaN/infinite/zero scale and bias.<span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance); [square root glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); [rounding glossary](../../../../glossary.md#term-rounding); [finite glossary](../../../../glossary.md#term-finite); [subnormals glossary](../../../../glossary.md#term-subnormal); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [NaN glossary](../../../../glossary.md#term-nan); [bias glossary](../../../../glossary.md#term-bias))</span>
- An independent StrictMath/high-precision oracle checks ordinary finite results within four
  result ulps for FLOAT64, two result ulps for FLOAT32, and one represented BFLOAT16 ulp.
  Exceptional classes, infinities, and zero signs use class/raw-sign assertions. The generated
  body and frozen identical-algorithm direct Java oracle are raw-bit equal.<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Resource declarations contain each unique logical input once in first-occurrence order and one
  output. They contain no workspace, materialization, partial/combine value, saved statistic,
  denominator cache, or hidden scratch.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [saved statistic glossary](../../../../glossary.md#saved-statistic); [cache glossary](../../../../glossary.md#term-cache))</span>
- Empty output performs no reads, square root, write, generated call, or worker submission and
  leaves all carriers/canaries unchanged.
- Cold binding validates exact carrier type, liveness/thread access, native order, alignment,
  complete accessed spans, output writability/injectivity, and every output/input overlap before
  mutation/submission. Failure leaves inputs, output, and canaries unchanged and submits no work.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [liveness glossary](../../../../glossary.md#term-liveness); [alignment glossary](../../../../glossary.md#term-alignment); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Scalar and parallel-scalar results are raw-bit deterministic across legal channel or
  non-channel ranges, worker counts, repeated runs, and concurrent runs. Scalar execution performs
  exactly <span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar))</span>`channelCount` square roots; parallel execution performs no more than
  `channelCount * submittedRangeCount`. Each nonempty output coordinate is written exactly once;
  input/input alias remains legal read sharing.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Generated classes are deterministic, final, field-free, constructor-free, and expose one public
  typed static `invoke` with concrete unique input/output carriers, packed primitive geometry, and
  `long start/end`. There is no scratch, `Object`, bridge descriptor, kind/form argument, or
  validation/reference call.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [validation glossary](../../../../glossary.md#term-validation))</span>
- Complete `javap -c -p` and `javap -v -p` inspection proves channel reads/root outside the
  selected coordinate loop, direct per-coordinate formula order, generation-time
  type/axis/carrier specialization, inline typed
  addressing, and one final store. It finds no allocation, boxing, reflection, method handles,<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
  `invokedynamic`, dynamic constants, bootstrap methods, collection/map/string lookup, semantic
  dispatch, or avoidable coordinate/address recomputation.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-method-handle-lookup))</span><span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- A reviewed complete member allowlist permits only required primitive/raw-bit operations,
  `Math.sqrt`, and typed [`MemorySegment`](../../../../glossary.md#term-memorysegment) access. Any unexpected member is a failure; generated
  classes reference no Synaptik numerical, validation, or reference helper.
- Specialization/cache identity distinguishes family, semantic input/result types, exact epsilon
  bits, rank/channel axis, algorithm/arithmetic version, boundary map, carrier/access pattern, and
  selected range form.
  Compatible extents/strides, bases, offsets, ranges, slots, workers, graph/run identities, and
  values do not change identity. Two epsilon bit patterns are incompatible specializations.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Schema advances exactly 48 to 49. Schema-48 and earlier envelopes safely miss; prior retained
  evidence/resources and completed task records remain immutable and historical.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Freeze exactly the following eight targets. Dense layouts have offset zero and canonical
  row-major strides. Carrier columns list<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major))</span>
  `input/scale/bias/runningMean/runningVariance -> output`; `S` means a matching native-order
  `MemorySegment`, primitive-array names are exact, and [vector](../../../../glossary.md#term-vector)-[layout](../../../../glossary.md#layout) pairs [mean](../../../../glossary.md#term-mean)
  `(offset, stride)`. Every [target](../../../../glossary.md#term-target) uses usable parallelism four and
  `minimumElementsPerWorker = 4096`, so the selected range form and four submitted ranges are
  deterministic.

  | Target | [Shape](../../../../glossary.md#shape) / [axis](../../../../glossary.md#term-axis) | Semantic input types -> result | Carriers | Layout details | Range form |
  |---|---|---|---|---|---|
  | `BN-BF16-A1` | `[32,64,256]` / 1 | all [BFLOAT16](../../../../glossary.md#term-bfloat16) -> BFLOAT16 | all `short[]` | dense | `NON_CHANNEL_RANGE` |
  | `BN-F32-A1` | `[32,64,256]` / 1 | all [FLOAT32](../../../../glossary.md#term-float32) -> FLOAT32 | all `float[]` | dense | `NON_CHANNEL_RANGE` |
  | `BN-F64-A1` | `[32,64,256]` / 1 | all [FLOAT64](../../../../glossary.md#term-float64) -> FLOAT64 | all `double[]` | dense | `NON_CHANNEL_RANGE` |
  | `BN-F32-A0` | `[4096,128]` / 0 | all FLOAT32 -> FLOAT32 | all `float[]` | dense | `CHANNEL_RANGE` |
  | `BN-F32-A2` | `[32,256,64]` / 2 | all FLOAT32 -> FLOAT32 | all `float[]` | dense | `NON_CHANNEL_RANGE` |
  | `BN-MIX-F64` | `[16,32,64]` / 1 | FLOAT32/FLOAT64/BFLOAT16/FLOAT32/FLOAT64 -> FLOAT64 | `float[]/S/short[]/S/double[] -> S` | input `(offset 11; strides [5000,137,2])`; vectors `(3,2)/(5,0)/(7,3)/(11,1)`; output `(13; [6000,151,2])` | `NON_CHANNEL_RANGE` |
  | `BN-MIX-F32` | `[16,32,64]` / 1 | BFLOAT16/FLOAT32/BFLOAT16/FLOAT32/BFLOAT16 -> FLOAT32 | `S/float[]/S/float[]/short[] -> float[]` | input `(offset 9; strides [5100,139,2])`; vectors `(2,2)/(4,0)/(6,3)/(10,1)`; output `(17; [6100,157,2])` | `NON_CHANNEL_RANGE` |
  | `BN-REPEAT-C1` | `[65536,1]` / 1 | all FLOAT32 -> FLOAT32 | `float[]/one repeated float[] boundary -> float[]` | dense; positions 1–4 share one exact `ValueId` | `NON_CHANNEL_RANGE` |

  The axis-last and `BN-REPEAT-C1` rows expose respectively the former per-prefix-root defect and
  useful small-channel parallelism; `BN-F32-A0` exercises channel-owned parallel ranges. Freeze
  unchanged FLOAT32 Layer Norm, RMS Norm, Softmax, and pointwise ADD controls at their existing
  retained CPU-0007F/0007E/0005 matrix definitions rather than silently redefining them.<span data-glossary-navigation="true"> ([Norm glossary](../../../../glossary.md#term-norm); [Softmax glossary](../../../../glossary.md#softmax--log-softmax); [pointwise glossary](../../../../glossary.md#term-pointwise); [CPU glossary](../../../../glossary.md#term-cpu); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Every target and control passes generated/direct `<= 1.15x` in each of five isolated forks and
  for the median of fork medians. The frozen optimal clean Java oracle performs identical
  validation, selected range traversal, useful arithmetic, addressing, conversions, and stores
  and is never slowed, padded, or routed through production helpers. A deliberately identical
  per-prefix or per-coordinate-root oracle is invalid even if the generated/direct ratio passes.<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- Focused tests, one final uncached CPU suite, Javadoc, retained semantic/Class-File/
  decompilation/member/checksum/five-fork evidence, exact scope, schema/status, staging, and
  whitespace gates pass. A distinct clean documentation context finalizes affected Javadocs, CPU
  guide, glossary impact, task, master plan, and roadmap without repeating stable Java/performance
  work.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [decompilation glossary](../../../../glossary.md#term-decompilation); [checksum glossary](../../../../glossary.md#term-checksum); [scope glossary](../../../../glossary.md#term-scope))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

The implementation context runs focused tests after stabilization, then one final [CPU](../../../../glossary.md#term-cpu) suite:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormInferenceGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuBatchNormInferenceReferenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest
./gradlew :backends:cpu:test --rerun-tasks
```

Retain one [immutable](../../../../glossary.md#term-immutable) evidence bundle outside the repository containing exact source/probe inputs,
environment/[JVM](../../../../glossary.md#term-java-virtual-machine-jvm) facts, all eight [targets](../../../../glossary.md#term-target) and four controls, specializations, complete
`javap -c -p` and `javap -v -p`, member reports, semantic/accuracy/range/[alias](../../../../glossary.md#aliasing-and-arena)/[canary](../../../../glossary.md#term-canary) results,
five isolated `-Xms1g -Xmx1g` [fork](../../../../glossary.md#term-fork-meanings) outputs, rejected whole [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), summaries, inventory, and a
[SHA-256](../../../../glossary.md#term-sha-256) [manifest](../../../../glossary.md#term-manifest). Each fork uses at least five randomized [warmup](../../../../glossary.md#term-warmup) [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), nine randomized measured
rounds, adaptive batches of at least 25 ms, deterministic inputs, randomized generated/direct
order, and raw/[checksum](../../../../glossary.md#term-checksum) verification. Reject and retain a whole fork if any target, control,
checksum, semantic, accuracy, environment, resource, or [scope](../../../../glossary.md#term-scope) gate fails.

The documentation-focused context receives the stabilized diff, final CPU XML, and retained
evidence and reuses successful Java/performance evidence unless executable behavior changes or it
records a concrete stale-evidence risk. After final Javadocs it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also inspects rendered Javadocs and validates Markdown links/anchors/fences/newlines/whitespace,
exact path/type ceilings, [schema](../../../../glossary.md#term-schema) 49 and historical-evidence wording, CPU 0007A1D Review needed,
CPU 0007F Complete, CPU 0007F1 Complete only after every gate, CPU 0007F2 and later tasks Draft
without detailed specifications, and empty staging.

Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration suites remain deferred to CPU
0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci) because the planned implementation changes one [concrete backend](../../../../glossary.md#concrete-backend)
without a shared boundary. Reaching a stop condition that changes a shared boundary requires
replanning and proportionate validation instead.

Validation tiers are explicit: focused tests plus one final uncached CPU [module](../../../../glossary.md#term-module) suite are task
validation; the clean documentation/[Javadoc](../../../../glossary.md#term-javadoc) pass closes the same task; repository-wide,
architecture, backend-conformance, and integration validation is the CPU 0009 capability
[checkpoint](../../../../glossary.md#term-checkpoint) or continuous-integration tier.

## Dependencies

- Complete Model 0021B owns the exact first-class inference kind, five ordered roles, arbitrary
  normalized channel axis, rank-one vector Shapes, direct running-variance formula, epsilon,
  ordered promotion, computation format, result metadata, empty behavior, and special values.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [inference glossary](../../../../glossary.md#term-inference); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [vector glossary](../../../../glossary.md#term-vector); [Shapes glossary](../../../../glossary.md#shape); [variance glossary](../../../../glossary.md#term-variance); [epsilon glossary](../../../../glossary.md#epsilon); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Complete [Compiler](../../../../glossary.md#term-compiler) 0005B and current `ReductionNormalizationInference` own final descriptor and
  channel-vector constraint validation. <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation))</span>`NormalizationGradientRules` owns gradients through
  ordinary Tensor expressions and requires no CPU forward contract change.<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [Tensor glossary](../../../../glossary.md#tensor); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Complete CPU 0007F supplies mixed-floating typed carriers, exact epsilon identity,
  first-occurrence boundary deduplication, static normalization layout handling, scalar/
  parallel-scalar orchestration, Class-File/member evidence, and the schema-48 frontier. Its
  trailing-slice IR, Layer scratch, and Layer/RMS numerical emitters are not reused as batch
  inference semantics.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [normalization glossary](../../../../glossary.md#term-normalization); [layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [schema glossary](../../../../glossary.md#term-schema); [slice glossary](../../../../glossary.md#slice); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Existing shared Prepare and Runtime contracts carry unique input/output boundaries and a
  zero-workspace direct invocation unchanged. CPU 0007A1D remains historical Review needed and is
  not a dependency.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [carry glossary](../../../../glossary.md#term-carry); [dependency glossary](../../../../glossary.md#term-dependency))</span>

All dependencies and forward semantics are settled. No architecture or shared-contract blocker is
known within this [fail-closed](../../../../glossary.md#term-fail-closed) static [scope](../../../../glossary.md#term-scope).

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0007F2 remains Draft after 0007F1 for five-input/five-output
  `BATCH_NORM_TRAINING`, including non-channel reductions, biased forward variance, unbiased
  running-variance transition, momentum, saved statistics, mixed output Shapes, truthful
  workspace, and exact multi-output binding. It receives no detailed specification here.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [variance glossary](../../../../glossary.md#term-variance); [momentum glossary](../../../../glossary.md#term-momentum); [Shapes glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding))</span>
- CPU 0008 follows CPU 0007F2 for remaining heavy portable families. CPU 0008B must preserve the
  prohibition on recognizing decomposed normalization as a first-class semantic kernel.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [kernel glossary](../../../../glossary.md#kernel))</span>
- CPU 0009 or continuous integration retains repository-wide portable coverage, conformance, and
  integration closure.<span data-glossary-navigation="true"> ([continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- CPU 0007A1D remains historical Review needed; its failed local evidence gate is independent.

## Architecture impact

Expected impact: None.

This task adds one [CPU](../../../../glossary.md#term-cpu)-private generated forward family behind unchanged [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare),
and [Runtime](../../../../glossary.md#term-runtime) contracts. If implementation requires a public/shared numerical contract, [dependency](../../../../glossary.md#term-dependency),
[module](../../../../glossary.md#term-module), resource kind, architecture rule, or build change, stop and report rather than editing
that boundary.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007F1. Work on the existing
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, CPU task 0007F1
and its dependencies, Model 0021B, the relevant Model batch-inference source contracts, Compiler
0005B and current normalization inference/gradient code, CPU 0007F, and current CPU capability/
IR/lowering/emitter/reference/route/prepare/finalization/executable/cache source and tests plus
documentation rules and applicable profiles.

Implement exactly CPU 0007F1 within its 46-path/four-type ceilings. Admit only explicit
BATCH_NORM_INFERENCE with the exact five ordered inputs, arbitrary normalized channel axis,
ordered floating promotion, typed epsilon, direct running variance, zero workspace, deterministic
channel/non-channel range selection, and schema 48-to-49 identity. Add the dedicated IR, lowerer,
direct emitter, and independent reference owner. Preserve validation-before-mutation, unique-boundary
declarations, direct generated execution, and fail-closed exclusions. Generated bytecode must
match the frozen optimal clean Java algorithm, selected range/hot-loop/dataflow shape, arithmetic
boundaries, and avoidable-overhead profile. Satisfy every semantic, accuracy, resource, alias,
Class-File/member, checksum, control, and five-fork performance gate without weakening or slowing
the oracle.

Stop on any ceiling, architecture uncertainty, semantic conflict, shared resource change,
BATCH_NORM_TRAINING work, or scope conflict. Hand the stabilized diff, final CPU XML, and retained
evidence to a distinct clean documentation-focused context to finalize Javadocs, CPU guide,
glossary, and planning in the same overall change. Do not mark Complete until all gates pass. Do
not commit, push, or stage.
```

## Documentation-focused finalization requirement

After executable Java and tests stabilize, a distinct clean documentation-focused agent/thread
must inspect the actual diff and retained evidence. It must apply General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend)
Guide, [Planning](../../../../glossary.md#term-planning), and Example profiles as relevant; finalize all affected Javadocs, the [CPU](../../../../glossary.md#term-cpu) guide,
glossary impact, this task, master plan, and roadmap; inspect rendered Javadocs; reuse successful
Java/performance evidence unless executable behavior changes; and record reasoned no-change
conclusions for [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) contracts, architecture/tests,
shared [Prepare](../../../../glossary.md#prepare)/backend-contract, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), and unrelated [modules](../../../../glossary.md#term-module).

## Local decisions

- Use a dedicated batch-inference IR/lowerer/emitter/reference quartet. Arbitrary channel-axis
  channel/non-channel geometry and zero resources are distinct from trailing Layer/RMS
  slice/reduction ownership.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [inference glossary](../../../../glossary.md#term-inference); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Scalar ranges own channels and compute one denominator per channel. Parallel preparation chooses
  the larger channel or flattened non-channel range domain, caps the existing bounded range count
  to that domain, and bakes the selected form. Non-channel ranges repeat a denominator at most
  once per channel per submitted range, preserving small-channel parallelism without the original
  per-prefix or per-coordinate square-root defect.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Reject both simpler extremes as the universal oracle. Per-coordinate and prefix-channel-slab
  traversal preserve dense row-major locality but repeat <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [row-major glossary](../../../../glossary.md#term-row-major))</span>`sqrt` by output or prefix count. Pure
  channel-major traversal minimizes roots but can underuse workers for small `channelCount` and
  repeatedly sweep cache-unfriendly channel-last/general layouts. The selected two-form design
  keeps scalar channel work minimal and bounds the parallel repetition by submitted ranges while
  assigning the larger independent domain to workers. It adds no workspace, materialized
  denominator table, runtime tuning, measured candidate set, or speculative 0007F2 mechanism.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [layouts glossary](../../../../glossary.md#layout); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Preserve the exact centered/divide/multiply/add formula rather than rewriting to channel
  coefficients. This retains Model 0021B's exceptional and signed-zero classes.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Keep concrete extents and strides cold. Rank, channel axis, semantic types/map, epsilon, access
  pattern, carriers, and algorithm version shape generated bytes and compatibility.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [Rank glossary](../../../../glossary.md#term-rank); [epsilon glossary](../../../../glossary.md#epsilon); [shape glossary](../../../../glossary.md#shape))</span>
- Advance generated compatibility exactly from schema 48 to 49; no source-format or persistent-
  envelope migration is required because older envelopes safely miss.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>

## Known limitations

- Only fully static resolved-layout scalar/parallel-scalar batch-inference forward execution is
  planned. Dynamic Shapes, Vector/native routes, fusion, materialization, and training remain
  unsupported.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [inference glossary](../../../../glossary.md#term-inference); [Shapes glossary](../../../../glossary.md#shape); [Vector glossary](../../../../glossary.md#term-vector); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization))</span>
- Finite results follow the selected CPU operation order and tolerance table. Raw-bit identity is
  required only between generated and identical-algorithm direct Java, not across backends.<span data-glossary-navigation="true"> ([Finite glossary](../../../../glossary.md#term-finite); [CPU glossary](../../../../glossary.md#term-cpu); [operation glossary](../../../../glossary.md#operation); [tolerance glossary](../../../../glossary.md#term-tolerance); [identity glossary](../../../../glossary.md#term-identity-meanings); [backends glossary](../../../../glossary.md#backend))</span>
- The performance gate covers only the frozen bounded matrix and recorded environment; it is not a
  general tuning or throughput claim.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [throughput glossary](../../../../glossary.md#term-throughput))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation and clean documentation review completed on 2026-08-25. Before implementation,
critical review replaced the original prefix-[channel](../../../../glossary.md#term-channel) slab plan because it repeated channel reads
and `sqrt` once per prefix, including `N` times per channel for dense `[N,C]` with a last channel
[axis](../../../../glossary.md#term-axis). The first generated body then still used per-[element](../../../../glossary.md#term-element) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) division/remainder and
avoidable address reconstruction and did not satisfy the frozen performance [shape](../../../../glossary.md#shape). The accepted
body decodes each range entry once, advances coordinates and addresses with an odometer and
increments, and retains genuine [FLOAT32](../../../../glossary.md#term-float32) locals for [BFLOAT16](../../../../glossary.md#term-bfloat16)/FLOAT32 computation. This evidence
does not claim universal [layout](../../../../glossary.md#layout) optimality.

Focused tests reported 112 passing cases. The first uncached [CPU](../../../../glossary.md#term-cpu)-suite attempt found two test-only
stale expectations: the internal-package inventory omitted the four newly authorized types, and
the regenerated [cache](../../../../glossary.md#term-cache)-envelope test still expected [schema](../../../../glossary.md#term-schema) 48. After correcting those tests, the
required full uncached rerun is retained as 462 tests, zero failures, zero errors, and two expected
skips. No successful Java or performance suite was repeated by the documentation context because
no executable Java behavior changed.

The [immutable](../../../../glossary.md#term-immutable) evidence root is
`/private/tmp/synaptik-cpu-0007f1-retained-evidence-20260825`. It contains 281 files. [SHA-256](../../../../glossary.md#term-sha-256) of
`SHA256SUMS` is
`185ecb1b1da84d20774b5f21979bbfc8cedb765cf03dc05610fa354bd7555029`, and every [manifest](../../../../glossary.md#term-manifest) entry
verifies. Its XML summary is `tests=462 failures=0 errors=0 skipped=2`; [specialization](../../../../glossary.md#term-specialization),
[decompilation](../../../../glossary.md#term-decompilation), verbose/member, [operation](../../../../glossary.md#operation)-count, semantic, [alias](../../../../glossary.md#aliasing-and-arena)/[canary](../../../../glossary.md#term-canary), accepted/rejected [fork](../../../../glossary.md#term-fork-meanings),
environment, and source records are present.

All eight [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [targets](../../../../glossary.md#term-target) and unchanged controls passed each of five accepted isolated forks and
the [median](../../../../glossary.md#term-median)-of-fork-medians gate at `<= 1.15x`. The tightest batch fork is BN-MIX-[F32](../../../../glossary.md#floating-point-precision-and-rounding) at
`1.149753751x`. Three rejected BN-MIX-F32 probe protocols remain rejected and are not counted as
evidence: a polymorphic wrapper (`1.262671316x`), a non-isolated call site (`1.189308487x`), and a
100 ms whole [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) (`1.156348589x`); a 25 ms whole sample also measured `1.151696740x`. The
accepted probe isolates selected calls without [padding](../../../../glossary.md#padding) or slowing the [oracle](../../../../glossary.md#term-test-oracle). Generated classes
are reviewed as optimal-clean-Java equivalents for the frozen specialized cases, not byte-for-
byte `javac` output or guaranteed identical [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) assembly.

Complete Class-File/member inspection found deterministic final, field-free, constructor-free
classes with exactly one typed static entry, no scratch [parameter](../../../../glossary.md#term-parameter), and only the required typed
array/segment access, `Math.sqrt`, and BFLOAT16 conversion members. Operation counts confirm four
channel-value reads per generated square-root root and the selected channel/non-channel [ownership](../../../../glossary.md#term-ownership-meanings).
Schema is exactly 49 and older envelopes are safe misses.

The final repository [scope](../../../../glossary.md#term-scope) is within the 46-path ceiling. The pre-documentation implementation
diff has exactly 27 paths: 14 production paths and 13 tests. Exactly four new CPU-private
production types exist: `CpuBatchNormInferenceIr`, `CpuBatchNormInferenceLowering`,
`CpuBatchNormInferenceEmitter`, and `CpuBatchNormInferenceReferenceKernel`. Documentation adds
only the CPU guide, glossary, this task, CPU master plan, and roadmap. Staging remains empty.

The clean documentation pass applied the General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend) Guide, [Planning](../../../../glossary.md#term-planning), and
Example profiles. CPU Javadoc generation and rendered-page inspection pass; Markdown links,
anchors, fences, terminology, terminal newlines, status/order/[dependency](../../../../glossary.md#term-dependency) consistency, imports,
generated-member/reflection evidence, exact path/type/schema gates, staging checks, and whitespace
checks pass. CPU 0007A1D remains `Review needed`, CPU 0007F remains `Complete`, CPU 0007F1 is
`Complete`, and CPU 0007F2 remains the next defined `Draft` frontier.

## Implementation notes

The [CPU](../../../../glossary.md#term-cpu) capability provider now admits only the exact first-class static [inference](../../../../glossary.md#term-inference) subset. A
dedicated CPU-private [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/emitter/reference quartet preserves all five semantic positions,
deduplicates repeated logical boundaries in first-[occurrence](../../../../glossary.md#term-occurrence) order, derives arbitrary-[axis](../../../../glossary.md#term-axis)
prefix/[channel](../../../../glossary.md#term-channel)/suffix geometry, selects the preparation-time range form, and emits one direct
typed body. Preparation, finalization, portable routing, generated [artifact](../../../../glossary.md#term-artifact) compatibility, and
execution [carry](../../../../glossary.md#term-carry) the optional geometry without changing shared [Prepare](../../../../glossary.md#prepare) or [Runtime](../../../../glossary.md#term-runtime) contracts.

The implementation accepts independent [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) inputs, exact occurrence-order
promotion, result-typed [epsilon](../../../../glossary.md#epsilon) bits, dense/broadcast/strided resolved [layouts](../../../../glossary.md#layout), typed [heap](../../../../glossary.md#term-heap) arrays,
native-order segments, and mixed carriers. It hoists scale, [bias](../../../../glossary.md#term-bias), running [mean](../../../../glossary.md#term-mean), running [variance](../../../../glossary.md#term-variance),
and one denominator per covered channel; evaluates centered/divide/multiply/add in the specified
order; uses FLOAT32 computation for BFLOAT16/FLOAT32 and FLOAT64 for FLOAT64; and stores directly.
It declares zero workspace/[materialization](../../../../glossary.md#materialization)/saved state, permits input/input [aliasing](../../../../glossary.md#term-aliasing), and rejects
output/input overlap before writes or submission. Training, [fusion](../../../../glossary.md#term-fusion), [vector](../../../../glossary.md#term-vector)/native paths, dynamic
[Shapes](../../../../glossary.md#shape), and [autotuning](../../../../glossary.md#autotuning--model-autotuning) remain excluded.

No change is required in [Tensor](../../../../glossary.md#tensor), [Compile](../../../../glossary.md#compile), or Training APIs; [Model](../../../../glossary.md#term-model) 0021B or [Compiler](../../../../glossary.md#term-compiler) 0005B
contracts; `ARCHITECTURE.md`, the current architecture plan, ADRs, or architecture tests; shared
Prepare/[backend](../../../../glossary.md#backend)-contract or Runtime APIs; backend-[conformance](../../../../glossary.md#term-conformance) or integration tests; [Gradle](../../../../glossary.md#term-gradle)/build
configuration; Vector/native routes; or unrelated [modules](../../../../glossary.md#term-module). Those contracts either already define
the semantics consumed here or remain outside this CPU-private realization.

## Completion summary

- Completed changes: exact static first-class CPU batch-normalization inference with arbitrary
  channel axis, ordered floating promotion, typed epsilon, deterministic dual range forms, direct
  generated execution, zero resources, and schema 49.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [inference glossary](../../../../glossary.md#term-inference); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [epsilon glossary](../../../../glossary.md#epsilon); [schema glossary](../../../../glossary.md#term-schema))</span>
- Files changed or created: 14 production paths, 13 test paths, and five documentation/planning
  paths; exactly four new CPU-private production types; total scope at or below 46 paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [scope glossary](../../../../glossary.md#term-scope))</span>
- Tests and validation: focused 112 passing; final uncached CPU XML 462 tests, zero failures, zero
  errors, two skips; CPU Javadoc successful; task-specific Markdown and exact repository gates
  successful; retained 281-file manifest fully verified.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [manifest glossary](../../../../glossary.md#term-manifest))</span>
- Documentation-agent review: mandatory independent clean documentation context completed without
  changing executable Java behavior; clean context/session identifier
  `01a03a19-dfac-7a61-90f5-0654784e9c74`.
- Documentation impact: CPU backend guide, glossary, this task, CPU master plan, and roadmap
  finalized; no architecture, public API, shared-contract, build, conformance, integration,
  Vector/native, or unrelated-module documentation change is required.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance); [Vector glossary](../../../../glossary.md#term-vector); [module glossary](../../../../glossary.md#term-module))</span>
- Javadoc review: all 14 changed production paths inspected; affected IR, lowering geometry,
  emitter, reference, portable-family, plan, lowering-result, and prepared-executable contracts
  finalized with ownership, ranges, parameters, results, and failures.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [parameters glossary](../../../../glossary.md#term-parameter))</span>
- Glossary impact: clarified only the current CPU inference realization and schema-49 artifact
  boundary without duplicating Model training semantics.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [Model glossary](../../../../glossary.md#term-model))</span>
- Unresolved issues: none for [CPU](../../../../glossary.md#term-cpu) 0007F1.
- Follow-up required: none for CPU 0007F1; CPU 0007F2 remains the separate Draft training frontier.

Status: Complete

## Status gate

Do not change the task, [CPU](../../../../glossary.md#term-cpu) master plan, or roadmap to `Complete` until every acceptance criterion,
the final uncached CPU suite, retained semantic/Class-File/member/performance evidence, final
[Javadoc](../../../../glossary.md#term-javadoc) and documentation [validation](../../../../glossary.md#term-validation), exact [scope](../../../../glossary.md#term-scope)/[schema](../../../../glossary.md#term-schema)/status checks, independent clean
documentation pass, and completion summary have passed. Any missing gate leaves the task
`In progress`, `Review needed`, or `Blocked` with the exact reason.
