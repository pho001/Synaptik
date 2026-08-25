# Task 0007F1: Portable Batch-Normalization Inference Coverage

## Status

Complete

## Goal

Execute exactly one fully static, resolved-layout, first-class `BATCH_NORM_INFERENCE` occurrence
on the portable CPU route. Preserve Model 0021B's stateless five-input/one-output semantics and
Compiler 0005B's validated first-class form through one direct generated coordinatewise body with
arbitrary normalized channel axis, exact ordered floating promotion, exact typed epsilon, explicit
affine and running-statistic reads, channel-hoisted square roots, deterministic scalar or
parallel-scalar ranges, and zero workspace.

## Mental model

```text
one explicit BATCH_NORM_INFERENCE node
  -> CPU validates one static input Shape and four exact [C] vectors
  -> CPU selects one channel-owned or non-channel-owned range form
  -> each invocation hoists channel values and one square root per channel it covers
  -> each coordinate executes the exact typed inference formula
  -> one direct generated store publishes the result
```

Scalar execution owns channels and computes each channel denominator exactly once. Parallel
execution partitions either channels or the flattened non-channel domain, whichever exposes the
larger bounded range domain. A non-channel range repeats each channel denominator at most once per
submitted range, not once per prefix or coordinate. Both forms keep the channel-axis contract
layout-neutral, require no cache or workspace, retain channel values in primitive locals across
stores so Java array-alias conservatism cannot force reloads, and give invocations disjoint output
regions.

## Scope

- Recognize exactly one compiled `BatchNormKind.BATCH_NORM_INFERENCE` occurrence with
  `BatchNormInferenceAttrs` and ordered
  `[input, scale, bias, runningMean, runningVariance] -> [output]`.
- Preserve the first-class semantic identity. Admit no training occurrence and never infer,
  reconstruct, fuse, or relabel batch normalization from decomposed arithmetic or reduction
  graphs.
- Accept BFLOAT16, FLOAT32, and FLOAT64 independently at all five input positions. Promote in
  exact occurrence order: input with scale, then bias, then running mean, then running variance.
  Require output and epsilon to use the resulting type and require output gradient eligibility to
  equal the logical OR of all five input descriptors.
- Require fully static resolved descriptors, input rank at least two, a normalized non-negative
  channel axis within rank, four exact rank-one vectors whose extent equals the selected channel
  extent, exact input/output Shape equality, non-negative checked layouts, and an injective
  writable output.
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
- Select exactly one range form during CPU preparation, before artifact realization:

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
  half-open channel interval and traverses the complete non-channel domain for each owned channel.
  `NON_CHANNEL_RANGE` interprets them as a half-open flattened non-channel interval and traverses
  every channel for that owned interval. The range form is a CPU-private code-shaping
  specialization/cache fact. It is not a runtime choice, measured candidate, tuning decision,
  fixed-Shape variant, or additional generated artifact for one preparation. Range orchestration
  uses `minimumRangeItemsPerWorker`, so the existing minimum-work threshold remains measured in
  output coordinates rather than accidentally treating one channel or one non-channel coordinate
  as equal work.
- Treat channel extent zero or any zero non-channel extent as an empty output. Retain truthful
  input/output declarations, but perform no input read, channel-vector read, output write,
  generated invocation, square root, or worker submission.
- In `CHANNEL_RANGE`, for each owned channel read and promote `scale[c]`, `bias[c]`,
  `runningMean[c]`, and `runningVariance[c]` once, compute the denominator once, then traverse all
  non-channel coordinates in canonical prefix-major/suffix-major order. Scalar execution
  therefore performs exactly `channelCount` square roots.
- In `NON_CHANNEL_RANGE`, one invocation traverses channels in increasing order. For each channel
  it performs the four promoted channel reads and denominator calculation once, then traverses
  only its owned non-channel interval. Parallel execution therefore performs at most
  `channelCount * submittedRangeCount` square roots. It never performs one square root per prefix,
  suffix position, or output coordinate. The bounded repetition preserves useful parallelism for
  small channel counts while partitioning the larger non-channel domain for locality.
- Preserve the exact inference formula and operation order for each coordinate:

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
  affine rewrite, fused multiply-add, or `input * alpha + beta` reassociation.
- Treat all five reads and epsilon in FLOAT32 when the result is BFLOAT16 or FLOAT32, and in
  FLOAT64 when the result is FLOAT64. Round each subtraction, addition, square-root result,
  division, multiplication, and final addition at that computation-format boundary. Encode
  BFLOAT16 once at the final store; FLOAT32 and FLOAT64 use their matching primitive store.
- Preserve ordinary floating behavior from Model 0021B. Negative radicands and negative-infinite
  running variance produce NaN through square root; exact zero radicands permit zero denominators;
  positive-infinite variance produces a positive-infinite denominator; input/statistic/affine NaN,
  infinities, signed zeros, overflow, and underflow follow the stated operation order. No later
  affine zero may suppress an already produced NaN.
- Deduplicate exact repeated logical input `ValueId` values in first-occurrence order and retain an
  immutable semantic-position-to-boundary map. Declare each unique input once and exactly one
  output. Input/input alias is read sharing; output/input overlap is always rejected before any
  mutation or worker submission.
- After successful cold validation, each generated store publishes its coordinate directly to the
  ordinary output carrier. There is no shadow output, commit phase, or rollback. Model-defined
  non-finite arithmetic does not throw; an unrelated later generated/worker failure follows the
  existing execution contract and may expose already written coordinates.
- Support each unique boundary's exact typed heap array or native-order `MemorySegment` carrier:
  `short[]`, `float[]`, or `double[]` according to that boundary's own type. Support all
  heap/segment combinations, arbitrary legal offsets, positive or zero input strides, and an
  arbitrary injective output layout. Cold binding resolves concrete carriers once; generated hot
  work performs no storage discovery or carrier switch.
- Add one focused batch-inference IR, one focused arbitrary-axis lowerer, one direct emitter, and
  one independent reference owner. Reuse existing access-plan, carrier, route, preparation,
  finalization, executable, worker, artifact, and overlap seams without extending the trailing
  Layer/RMS IR or creating a generic normalization interpreter.
- Select scalar or parallel-scalar execution only. Use zero workspace, zero materialization, zero
  partial/combine state, and zero saved-statistic storage. Worker count and ranges remain cold
  invocation facts; generated code creates and submits no workers.
- Emit one deterministic final, field-free, constructor-free generated class per specialization
  with one public static typed `invoke` entry. Its exact parameters are the unique input carriers
  in first-occurrence order, the output carrier, one packed primitive `long[]` geometry, and
  primitive `long start, long end`. It has no scratch parameter.
- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly once from 48 to 49. The new family,
  channel-axis geometry, five-position boundary map, epsilon/formula identity, and generated body
  change artifact meaning. Schema 48 and earlier envelopes are safe incompatible misses; retained
  historical evidence remains immutable.
- Use a frozen optimal clean Java implementation with the same selected range form, validation,
  carrier and layout addressing, channel-hoisted reads/root, per-coordinate arithmetic
  boundaries, and stores as the design, review, and performance oracle. The oracle must be
  independently defensible: it may not repeat channel-only work per prefix or coordinate merely
  to mirror generated code. Generated code may call only required JDK primitive,
  raw-bit, `Math.sqrt`, and typed `MemorySegment` operations and must call no Synaptik numerical,
  validation, or reference helper.

## Out of scope

- `BATCH_NORM_TRAINING`, batch mean or variance reductions, biased or unbiased statistic
  calculation, momentum, next running statistics, saved mean, saved inverse standard deviation,
  statistic transitions, five-output binding, or any CPU 0007F2 work.
- Layer Norm, RMS Norm, softmax, group/instance/local-response/weight/spectral normalization,
  training/evaluation flags, module/parameter ownership, state mutation, backward execution, or
  gradient-policy changes.
- Optional affine/statistic operands, implicit ones or zeros, scalar/broadcast/full-rank channel
  operands, multiple/inferred/default channel axes, NCHW/NHWC aliases, default or Tensor epsilon,
  or configurable output/computation type.
- Value-based rejection of negative running variance, variance repair, standard-deviation or
  inverse-standard-deviation input, denominator sentinel, relaxed math, or a cross-backend
  bitwise-finite-result promise.
- Decomposition, recognition, or fusion of equivalent graphs; fusion with adjacent work; Vector
  API or native/vendor routes; runtime kind/form dispatch; or a scalar-reference Runtime fallback.
- Dynamic or symbolic Shapes, unresolved layouts, runtime channel axis, rank-zero/rank-one input,
  FLOAT16, integral/BOOL/quantized/complex data, negative resolved strides, non-injective output,
  or any checked count/span/address that cannot be represented.
- Per-element task submission, per-coordinate square roots or channel-vector reloads, workspace,
  materialized channel vectors, precomputed channel-factor arrays, saved denominators, new
  resource kinds, or hidden caches.
- Generated calls to CPU validation/reference code; allocation, boxing, reflection, method
  handles, `invokedynamic`, dynamic constants, bootstrap methods, `Object` descriptors,
  collection/map/string lookup, or avoidable semantic dispatch in generated hot work.
- Public/shared API, Model, Compiler, Training, Runtime, Prepare, backend-contract, dependency,
  module-boundary, Gradle/toolchain, architecture, conformance, integration, trace, tuning, or
  persistence-format changes beyond the CPU-private generator schema.
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
  concrete arithmetic realization, resource/access validation, and generated execution.
- CPU analysis must select only explicit `BATCH_NORM_INFERENCE`. Neither CPU nor Compiler may
  infer a first-class normalization kind from an equivalent decomposed graph.
- The completed Model and Compiler contracts are implementation inputs, not files to revise. Any
  disagreement in signature, descriptor, channel geometry, promotion, epsilon, or formula is a
  stop condition.
- CPU analysis/lowering chooses the route and computes every static eligibility and exact resource
  fact before shared slot assignment. CPU finalization constructs or reuses the generated artifact
  only afterward. Cold binding validates carriers, thread access, spans, output injectivity, and
  overlap before mutation or worker submission. Runtime receives only a prepared executable.
- Planning continues to choose only CPU ownership. Shared Prepare remains CPU-blind and assigns
  declared buffer slots without interpreting channel axes, arithmetic, routes, or specialization.
- Generated code must preserve the frozen optimal clean Java algorithm, selected range/hot-loop/dataflow
  shape, arithmetic boundaries, and avoidable-overhead profile. A reference call, generic bridge,
  per-element callback, packed runtime kind switch, pointwise decomposition, or formula rewrite is
  not a specialized implementation.
- Any required public/shared/build/architecture/conformance/integration change, new resource kind,
  workspace, materialization, unresolved semantic decision, or training/statistic-transition work
  is a stop/replan condition.

## Current-code evidence

- `BatchNormKind` fixes inference to `BatchNormInferenceAttrs`, five inputs, and one output;
  `TensorBatchNormInferenceExpressions` fixes ordered roles, rank/vector Shape rules, promotion,
  exact epsilon type, result Shape, gradient eligibility, and provenance.
- `ReductionNormalizationInference` revalidates the normalized axis, four channel-vector
  equalities, ordered promotion, exact result descriptor, and epsilon. It leaves the occurrence
  first-class. `NormalizationGradientRules` consumes that exact occurrence for gradients but does
  not define CPU forward execution.
- `CpuCapabilityProvider` currently admits Softmax and trailing Layer/RMS normalization only;
  `BATCH_NORM_INFERENCE` therefore fails closed today.
- `CpuPortableKernelIr`, `CpuPartitionLowering`, `CpuPortableRoutePlan`,
  `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`, `CpuPartitionFinalizer`, and
  `CpuPreparedExecutable` enumerate focused family IR/geometry variants and already provide the
  needed one-unit analysis, exact declaration, cold carrier binding, range orchestration, overlap,
  finalization, and direct invocation seams.
- `CpuTrailingNormalizationIr` and `CpuTrailingNormalizationLowering` prove ordered mixed-type
  carriers, first-occurrence input deduplication, typed epsilon identity, static non-negative
  layout handling, output injectivity, and optional-family resource accounting. Batch inference
  needs a separate arbitrary-axis zero-workspace geometry rather than extending their trailing
  Shape and Layer-only scratch contract.
- `CpuClassFileKernelGenerator`, `CpuKernelSpecialization`, and `CpuGeneratorSchema` currently emit
  one typed direct `invoke` and identify schema 48. The family dispatch and public-entry allowlist
  do not yet include batch inference.
- Current generated evidence tests lock deterministic bytes, exact members, direct bodies,
  carrier combinations, schema compatibility, and five-fork generated/direct conventions. CPU
  0007F is the closest mixed-floating normalization precedent; no existing numerical emitter is a
  valid batch-inference implementation.

## Package impact

Existing CPU-private packages changed:

- `io.github.pho001.synaptik.backend.cpu` — exact truthful capability admission.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — immutable batch-inference identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — arbitrary-axis channel/non-channel geometry and
  exact unique-boundary derivation.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct typed formula body.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent differential oracle.
- Existing `route.portable`, `prepare`, `executable`, and `cache` packages — realization,
  declaration, cold validation/invocation, and schema-49 compatibility.

Packages added: None.

Type placement:

- Add `CpuBatchNormInferenceIr` in `internal.ir` for exact kind/form, five semantic input types,
  promoted result, epsilon raw bits, input rank, channel axis, algorithm/arithmetic version,
  semantic-position map, and access patterns.
- Add `CpuBatchNormInferenceLowering` in `internal.lowering` for exact static Shape/layout,
  prefix/channel/suffix/non-channel/output counts, canonical coordinate/address mapping,
  first-occurrence unique boundaries, checked spans, the selected range form, and zero-resource
  geometry.
- Add `CpuBatchNormInferenceEmitter` in `internal.codegen.emit` for the sole direct channel-hoisted
  and per-coordinate numerical body selected at generation time.
- Add `CpuBatchNormInferenceReferenceKernel` in `internal.reference`; it independently derives
  logical channel coordinates and formula expectations without production lowering, emission,
  binding, or generated helpers.
- Add no validator type because every represented running-variance and affine value has defined
  Model behavior and existing cold structural/overlap validation is sufficient. Add no facade,
  manager, registry, generic normalization interpreter, or change to Layer/RMS emitters.

## Affected files

Expected production/Javadoc paths:

- `CpuCapabilityProvider.java` and CPU package Javadoc;
- `internal/ir/CpuPortableKernelIr.java`, new `CpuBatchNormInferenceIr.java`, and IR package
  Javadoc;
- `internal/lowering/CpuPartitionLowering.java`, new `CpuBatchNormInferenceLowering.java`, and
  lowering package Javadoc;
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new
  `CpuBatchNormInferenceEmitter.java`, existing carrier/loop emission seams only where required,
  and emitter package Javadoc;
- `internal/cache/CpuGeneratorSchema.java`, `CpuKernelSpecialization.java` only if exact entry or
  compatibility validation requires it, and cache package Javadoc;
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
  and reference-differential tests only where their exact inventories or behavior change; and
- the historical pointwise-ledger test only for its current-schema assertion; retained historical
  evidence resources must not change.

Expected documentation/planning paths:

- `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  global roadmap. Architecture, public Tensor/Compile/Training APIs, Model/Compiler plans, shared
  Prepare, backend-contract, conformance, integration, and build files are review-only unless a
  stop condition is reached.

## Maximum scope

The complete implementation, tests, and documentation may modify or create at most 46 repository
paths: at most 27 production/Javadoc paths, 14 test paths, and exactly the five named
documentation/planning paths. Exactly four new CPU-private production types are permitted: the
IR, lowerer, emitter, and independent reference owner listed above.

This exceeds the planning guide's ordinary file-count guardrail because one truthful CPU family
must update existing sealed IR, lowering, generator, schema/cache, route, prepare, executable,
package-Javadoc, inventory, and differential seams atomically. The behavioral core remains four
focused types in one module. Splitting those integrations would temporarily advertise, prepare,
or persist a family that cannot execute and would duplicate the final module/evidence gates.

A 47th path, fifth production type, second emitter/validator, generic normalization owner,
Layer/RMS semantic change, public/shared/build/architecture/conformance/integration path,
workspace, materialization, or CPU 0007F2 behavior is a stop/replan condition. An omitted expected
path does not authorize an unrelated path.

## Detailed generated-code design boundary

- `CpuBatchNormInferenceIr` is instruction-free and exposes one deterministic family identity.
  Identity includes exact semantic input order/types, result type, epsilon raw bits, input rank,
  normalized channel axis, algorithm/arithmetic version, semantic-position map, and boundary
  access plans. Concrete extents, strides, offsets, carrier instances, slots, ranges, workers,
  graph/run identities, and Tensor values remain cold.
- `CpuBatchNormInferenceLowering.Geometry` retains resolved layouts for unique inputs and output,
  prefix/channel/suffix/non-channel/output counts, channel-axis mapping, and packs only primitive
  bases and layout geometry. CPU preparation adds the selected `CHANNEL_RANGE` or
  `NON_CHANNEL_RANGE` form. Compatible concrete extents and strides do not enter class identity
  because emitted code reads them from geometry; no fixed-Shape specialization is authorized.
- In either generated form, the channel loop loads the four channel values and computes the typed
  denominator before its coordinate loop. The coordinate loop advances primitive input/output
  coordinates and addresses directly, loads one input, performs the six exact typed arithmetic
  steps, and stores one output. General layouts use typed primitive odometers; proved dense
  heap-array forms may use cold-proved `int` indexes. No hot operation, carrier, layout,
  channel-axis, or range-form switch remains.
- The emitter bakes exact epsilon bits, result computation format, channel axis, semantic boundary
  map, access form, and selected range form. It does not bake concrete carrier bases, extents,
  strides, offsets, start/end, workers, or values.
- One specialization has exactly one generated artifact. The existing four-candidate/one-artifact/
  zero-shape-variant/zero-unroll-variant budget remains unchanged. Batch inference is never
  materialized and is never Vector-selected in this task.
- Schema 49 is current-only. A schema-48 or earlier persistent envelope safely misses and may be
  regenerated; no migration reader, compatibility alias, or mutation of historical evidence is
  added.

## Acceptance criteria

- Capability admits exactly `BATCH_NORM_INFERENCE` with `BatchNormInferenceAttrs`, five ordered
  inputs, one output, all current BFLOAT16/FLOAT32/FLOAT64 promotion combinations, exact epsilon,
  rank/vector Shape rules, static resolved layouts, result descriptor, and injective output.
  `BATCH_NORM_TRAINING` and every excluded occurrence fail closed.
- Source-backed tests prove an equivalent decomposed graph remains ordinary operations and never
  becomes `CpuBatchNormInferenceIr`; Layer/RMS behavior and capability remain unchanged.
- Lowering records exact five-position types, result type, epsilon bits, rank/channel axis,
  prefix/channel/suffix/non-channel/output counts, first-occurrence unique-boundary map, access
  facts, and zero-resource identity. Preparation records the exact deterministic range form and
  bounded range count. Checked count/span/address arithmetic precedes declarations.
- Semantic/differential coverage includes channel axis first, middle, and last; rank two and
  higher; channel extent one, larger, and zero; zero extents before and after the channel;
  same-type and mixed-floating inputs; dense/general/zero-stride reads; non-contiguous injective
  output; offsets; repeated vector `ValueId` inputs; and representative all-heap, all-segment, and
  mixed-carrier patterns.
- Formula tests cover direct running variance, epsilon inside square root, multiply-then-add
  affine order, and the exact computation-format rounding sequence. They include positive and
  negative finite variance, radicand negative/zero/positive, signed zeros, subnormals, raw NaNs,
  both infinities, overflow/underflow, and NaN/infinite/zero scale and bias.
- An independent StrictMath/high-precision oracle checks ordinary finite results within four
  result ulps for FLOAT64, two result ulps for FLOAT32, and one represented BFLOAT16 ulp.
  Exceptional classes, infinities, and zero signs use class/raw-sign assertions. The generated
  body and frozen identical-algorithm direct Java oracle are raw-bit equal.
- Resource declarations contain each unique logical input once in first-occurrence order and one
  output. They contain no workspace, materialization, partial/combine value, saved statistic,
  denominator cache, or hidden scratch.
- Empty output performs no reads, square root, write, generated call, or worker submission and
  leaves all carriers/canaries unchanged.
- Cold binding validates exact carrier type, liveness/thread access, native order, alignment,
  complete accessed spans, output writability/injectivity, and every output/input overlap before
  mutation/submission. Failure leaves inputs, output, and canaries unchanged and submits no work.
- Scalar and parallel-scalar results are raw-bit deterministic across legal channel or
  non-channel ranges, worker counts, repeated runs, and concurrent runs. Scalar execution performs
  exactly `channelCount` square roots; parallel execution performs no more than
  `channelCount * submittedRangeCount`. Each nonempty output coordinate is written exactly once;
  input/input alias remains legal read sharing.
- Generated classes are deterministic, final, field-free, constructor-free, and expose one public
  typed static `invoke` with concrete unique input/output carriers, packed primitive geometry, and
  `long start/end`. There is no scratch, `Object`, bridge descriptor, kind/form argument, or
  validation/reference call.
- Complete `javap -c -p` and `javap -v -p` inspection proves channel reads/root outside the
  selected coordinate loop, direct per-coordinate formula order, generation-time
  type/axis/carrier specialization, inline typed
  addressing, and one final store. It finds no allocation, boxing, reflection, method handles,
  `invokedynamic`, dynamic constants, bootstrap methods, collection/map/string lookup, semantic
  dispatch, or avoidable coordinate/address recomputation.
- A reviewed complete member allowlist permits only required primitive/raw-bit operations,
  `Math.sqrt`, and typed `MemorySegment` access. Any unexpected member is a failure; generated
  classes reference no Synaptik numerical, validation, or reference helper.
- Specialization/cache identity distinguishes family, semantic input/result types, exact epsilon
  bits, rank/channel axis, algorithm/arithmetic version, boundary map, carrier/access pattern, and
  selected range form.
  Compatible extents/strides, bases, offsets, ranges, slots, workers, graph/run identities, and
  values do not change identity. Two epsilon bit patterns are incompatible specializations.
- Schema advances exactly 48 to 49. Schema-48 and earlier envelopes safely miss; prior retained
  evidence/resources and completed task records remain immutable and historical.
- Freeze exactly the following eight targets. Dense layouts have offset zero and canonical
  row-major strides. Carrier columns list
  `input/scale/bias/runningMean/runningVariance -> output`; `S` means a matching native-order
  `MemorySegment`, primitive-array names are exact, and vector-layout pairs mean
  `(offset, stride)`. Every target uses usable parallelism four and
  `minimumElementsPerWorker = 4096`, so the selected range form and four submitted ranges are
  deterministic.

  | Target | Shape / axis | Semantic input types -> result | Carriers | Layout details | Range form |
  |---|---|---|---|---|---|
  | `BN-BF16-A1` | `[32,64,256]` / 1 | all BFLOAT16 -> BFLOAT16 | all `short[]` | dense | `NON_CHANNEL_RANGE` |
  | `BN-F32-A1` | `[32,64,256]` / 1 | all FLOAT32 -> FLOAT32 | all `float[]` | dense | `NON_CHANNEL_RANGE` |
  | `BN-F64-A1` | `[32,64,256]` / 1 | all FLOAT64 -> FLOAT64 | all `double[]` | dense | `NON_CHANNEL_RANGE` |
  | `BN-F32-A0` | `[4096,128]` / 0 | all FLOAT32 -> FLOAT32 | all `float[]` | dense | `CHANNEL_RANGE` |
  | `BN-F32-A2` | `[32,256,64]` / 2 | all FLOAT32 -> FLOAT32 | all `float[]` | dense | `NON_CHANNEL_RANGE` |
  | `BN-MIX-F64` | `[16,32,64]` / 1 | FLOAT32/FLOAT64/BFLOAT16/FLOAT32/FLOAT64 -> FLOAT64 | `float[]/S/short[]/S/double[] -> S` | input `(offset 11; strides [5000,137,2])`; vectors `(3,2)/(5,0)/(7,3)/(11,1)`; output `(13; [6000,151,2])` | `NON_CHANNEL_RANGE` |
  | `BN-MIX-F32` | `[16,32,64]` / 1 | BFLOAT16/FLOAT32/BFLOAT16/FLOAT32/BFLOAT16 -> FLOAT32 | `S/float[]/S/float[]/short[] -> float[]` | input `(offset 9; strides [5100,139,2])`; vectors `(2,2)/(4,0)/(6,3)/(10,1)`; output `(17; [6100,157,2])` | `NON_CHANNEL_RANGE` |
  | `BN-REPEAT-C1` | `[65536,1]` / 1 | all FLOAT32 -> FLOAT32 | `float[]/one repeated float[] boundary -> float[]` | dense; positions 1–4 share one exact `ValueId` | `NON_CHANNEL_RANGE` |

  The axis-last and `BN-REPEAT-C1` rows expose respectively the former per-prefix-root defect and
  useful small-channel parallelism; `BN-F32-A0` exercises channel-owned parallel ranges. Freeze
  unchanged FLOAT32 Layer Norm, RMS Norm, Softmax, and pointwise ADD controls at their existing
  retained CPU-0007F/0007E/0005 matrix definitions rather than silently redefining them.
- Every target and control passes generated/direct `<= 1.15x` in each of five isolated forks and
  for the median of fork medians. The frozen optimal clean Java oracle performs identical
  validation, selected range traversal, useful arithmetic, addressing, conversions, and stores
  and is never slowed, padded, or routed through production helpers. A deliberately identical
  per-prefix or per-coordinate-root oracle is invalid even if the generated/direct ratio passes.
- Focused tests, one final uncached CPU suite, Javadoc, retained semantic/Class-File/
  decompilation/member/checksum/five-fork evidence, exact scope, schema/status, staging, and
  whitespace gates pass. A distinct clean documentation context finalizes affected Javadocs, CPU
  guide, glossary impact, task, master plan, and roadmap without repeating stable Java/performance
  work.

## Tests / validation

The implementation context runs focused tests after stabilization, then one final CPU suite:

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

Retain one immutable evidence bundle outside the repository containing exact source/probe inputs,
environment/JVM facts, all eight targets and four controls, specializations, complete
`javap -c -p` and `javap -v -p`, member reports, semantic/accuracy/range/alias/canary results,
five isolated `-Xms1g -Xmx1g` fork outputs, rejected whole samples, summaries, inventory, and a
SHA-256 manifest. Each fork uses at least five randomized warmup batches, nine randomized measured
rounds, adaptive batches of at least 25 ms, deterministic inputs, randomized generated/direct
order, and raw/checksum verification. Reject and retain a whole fork if any target, control,
checksum, semantic, accuracy, environment, resource, or scope gate fails.

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
exact path/type ceilings, schema 49 and historical-evidence wording, CPU 0007A1D Review needed,
CPU 0007F Complete, CPU 0007F1 Complete only after every gate, CPU 0007F2 and later tasks Draft
without detailed specifications, and empty staging.

Repository-wide, architecture, backend-conformance, and integration suites remain deferred to CPU
0009 or continuous integration because the planned implementation changes one concrete backend
without a shared boundary. Reaching a stop condition that changes a shared boundary requires
replanning and proportionate validation instead.

Validation tiers are explicit: focused tests plus one final uncached CPU module suite are task
validation; the clean documentation/Javadoc pass closes the same task; repository-wide,
architecture, backend-conformance, and integration validation is the CPU 0009 capability
checkpoint or continuous-integration tier.

## Dependencies

- Complete Model 0021B owns the exact first-class inference kind, five ordered roles, arbitrary
  normalized channel axis, rank-one vector Shapes, direct running-variance formula, epsilon,
  ordered promotion, computation format, result metadata, empty behavior, and special values.
- Complete Compiler 0005B and current `ReductionNormalizationInference` own final descriptor and
  channel-vector constraint validation. `NormalizationGradientRules` owns gradients through
  ordinary Tensor expressions and requires no CPU forward contract change.
- Complete CPU 0007F supplies mixed-floating typed carriers, exact epsilon identity,
  first-occurrence boundary deduplication, static normalization layout handling, scalar/
  parallel-scalar orchestration, Class-File/member evidence, and the schema-48 frontier. Its
  trailing-slice IR, Layer scratch, and Layer/RMS numerical emitters are not reused as batch
  inference semantics.
- Existing shared Prepare and Runtime contracts carry unique input/output boundaries and a
  zero-workspace direct invocation unchanged. CPU 0007A1D remains historical Review needed and is
  not a dependency.

All dependencies and forward semantics are settled. No architecture or shared-contract blocker is
known within this fail-closed static scope.

## Follow-up tasks

- CPU 0007F2 remains Draft after 0007F1 for five-input/five-output
  `BATCH_NORM_TRAINING`, including non-channel reductions, biased forward variance, unbiased
  running-variance transition, momentum, saved statistics, mixed output Shapes, truthful
  workspace, and exact multi-output binding. It receives no detailed specification here.
- CPU 0008 follows CPU 0007F2 for remaining heavy portable families. CPU 0008B must preserve the
  prohibition on recognizing decomposed normalization as a first-class semantic kernel.
- CPU 0009 or continuous integration retains repository-wide portable coverage, conformance, and
  integration closure.
- CPU 0007A1D remains historical Review needed; its failed local evidence gate is independent.

## Architecture impact

Expected impact: None.

This task adds one CPU-private generated forward family behind unchanged Model, Compiler, Prepare,
and Runtime contracts. If implementation requires a public/shared numerical contract, dependency,
module, resource kind, architecture rule, or build change, stop and report rather than editing
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
must inspect the actual diff and retained evidence. It must apply General, API/Javadoc, Backend
Guide, Planning, and Example profiles as relevant; finalize all affected Javadocs, the CPU guide,
glossary impact, this task, master plan, and roadmap; inspect rendered Javadocs; reuse successful
Java/performance evidence unless executable behavior changes; and record reasoned no-change
conclusions for Tensor/Compile/Training APIs, Model/Compiler contracts, architecture/tests,
shared Prepare/backend-contract, conformance/integration, Gradle, and unrelated modules.

## Local decisions

- Use a dedicated batch-inference IR/lowerer/emitter/reference quartet. Arbitrary channel-axis
  channel/non-channel geometry and zero resources are distinct from trailing Layer/RMS
  slice/reduction ownership.
- Scalar ranges own channels and compute one denominator per channel. Parallel preparation chooses
  the larger channel or flattened non-channel range domain, caps the existing bounded range count
  to that domain, and bakes the selected form. Non-channel ranges repeat a denominator at most
  once per channel per submitted range, preserving small-channel parallelism without the original
  per-prefix or per-coordinate square-root defect.
- Reject both simpler extremes as the universal oracle. Per-coordinate and prefix-channel-slab
  traversal preserve dense row-major locality but repeat `sqrt` by output or prefix count. Pure
  channel-major traversal minimizes roots but can underuse workers for small `channelCount` and
  repeatedly sweep cache-unfriendly channel-last/general layouts. The selected two-form design
  keeps scalar channel work minimal and bounds the parallel repetition by submitted ranges while
  assigning the larger independent domain to workers. It adds no workspace, materialized
  denominator table, runtime tuning, measured candidate set, or speculative 0007F2 mechanism.
- Preserve the exact centered/divide/multiply/add formula rather than rewriting to channel
  coefficients. This retains Model 0021B's exceptional and signed-zero classes.
- Keep concrete extents and strides cold. Rank, channel axis, semantic types/map, epsilon, access
  pattern, carriers, and algorithm version shape generated bytes and compatibility.
- Advance generated compatibility exactly from schema 48 to 49; no source-format or persistent-
  envelope migration is required because older envelopes safely miss.

## Known limitations

- Only fully static resolved-layout scalar/parallel-scalar batch-inference forward execution is
  planned. Dynamic Shapes, Vector/native routes, fusion, materialization, and training remain
  unsupported.
- Finite results follow the selected CPU operation order and tolerance table. Raw-bit identity is
  required only between generated and identical-algorithm direct Java, not across backends.
- The performance gate covers only the frozen bounded matrix and recorded environment; it is not a
  general tuning or throughput claim.

## Validation evidence

Implementation and clean documentation review completed on 2026-08-25. Before implementation,
critical review replaced the original prefix-channel slab plan because it repeated channel reads
and `sqrt` once per prefix, including `N` times per channel for dense `[N,C]` with a last channel
axis. The first generated body then still used per-element coordinate division/remainder and
avoidable address reconstruction and did not satisfy the frozen performance shape. The accepted
body decodes each range entry once, advances coordinates and addresses with an odometer and
increments, and retains genuine FLOAT32 locals for BFLOAT16/FLOAT32 computation. This evidence
does not claim universal layout optimality.

Focused tests reported 112 passing cases. The first uncached CPU-suite attempt found two test-only
stale expectations: the internal-package inventory omitted the four newly authorized types, and
the regenerated cache-envelope test still expected schema 48. After correcting those tests, the
required full uncached rerun is retained as 462 tests, zero failures, zero errors, and two expected
skips. No successful Java or performance suite was repeated by the documentation context because
no executable Java behavior changed.

The immutable evidence root is
`/private/tmp/synaptik-cpu-0007f1-retained-evidence-20260825`. It contains 281 files. SHA-256 of
`SHA256SUMS` is
`185ecb1b1da84d20774b5f21979bbfc8cedb765cf03dc05610fa354bd7555029`, and every manifest entry
verifies. Its XML summary is `tests=462 failures=0 errors=0 skipped=2`; specialization,
decompilation, verbose/member, operation-count, semantic, alias/canary, accepted/rejected fork,
environment, and source records are present.

All eight batch targets and unchanged controls passed each of five accepted isolated forks and
the median-of-fork-medians gate at `<= 1.15x`. The tightest batch fork is BN-MIX-F32 at
`1.149753751x`. Three rejected BN-MIX-F32 probe protocols remain rejected and are not counted as
evidence: a polymorphic wrapper (`1.262671316x`), a non-isolated call site (`1.189308487x`), and a
100 ms whole sample (`1.156348589x`); a 25 ms whole sample also measured `1.151696740x`. The
accepted probe isolates selected calls without padding or slowing the oracle. Generated classes
are reviewed as optimal-clean-Java equivalents for the frozen specialized cases, not byte-for-
byte `javac` output or guaranteed identical JIT assembly.

Complete Class-File/member inspection found deterministic final, field-free, constructor-free
classes with exactly one typed static entry, no scratch parameter, and only the required typed
array/segment access, `Math.sqrt`, and BFLOAT16 conversion members. Operation counts confirm four
channel-value reads per generated square-root root and the selected channel/non-channel ownership.
Schema is exactly 49 and older envelopes are safe misses.

The final repository scope is within the 46-path ceiling. The pre-documentation implementation
diff has exactly 27 paths: 14 production paths and 13 tests. Exactly four new CPU-private
production types exist: `CpuBatchNormInferenceIr`, `CpuBatchNormInferenceLowering`,
`CpuBatchNormInferenceEmitter`, and `CpuBatchNormInferenceReferenceKernel`. Documentation adds
only the CPU guide, glossary, this task, CPU master plan, and roadmap. Staging remains empty.

The clean documentation pass applied the General, API/Javadoc, Backend Guide, Planning, and
Example profiles. CPU Javadoc generation and rendered-page inspection pass; Markdown links,
anchors, fences, terminology, terminal newlines, status/order/dependency consistency, imports,
generated-member/reflection evidence, exact path/type/schema gates, staging checks, and whitespace
checks pass. CPU 0007A1D remains `Review needed`, CPU 0007F remains `Complete`, CPU 0007F1 is
`Complete`, and CPU 0007F2 remains the next defined `Draft` frontier.

## Implementation notes

The CPU capability provider now admits only the exact first-class static inference subset. A
dedicated CPU-private IR/lowering/emitter/reference quartet preserves all five semantic positions,
deduplicates repeated logical boundaries in first-occurrence order, derives arbitrary-axis
prefix/channel/suffix geometry, selects the preparation-time range form, and emits one direct
typed body. Preparation, finalization, portable routing, generated artifact compatibility, and
execution carry the optional geometry without changing shared Prepare or Runtime contracts.

The implementation accepts independent BFLOAT16/FLOAT32/FLOAT64 inputs, exact occurrence-order
promotion, result-typed epsilon bits, dense/broadcast/strided resolved layouts, typed heap arrays,
native-order segments, and mixed carriers. It hoists scale, bias, running mean, running variance,
and one denominator per covered channel; evaluates centered/divide/multiply/add in the specified
order; uses FLOAT32 computation for BFLOAT16/FLOAT32 and FLOAT64 for FLOAT64; and stores directly.
It declares zero workspace/materialization/saved state, permits input/input aliasing, and rejects
output/input overlap before writes or submission. Training, fusion, vector/native paths, dynamic
Shapes, and autotuning remain excluded.

No change is required in Tensor, Compile, or Training APIs; Model 0021B or Compiler 0005B
contracts; `ARCHITECTURE.md`, the current architecture plan, ADRs, or architecture tests; shared
Prepare/backend-contract or Runtime APIs; backend-conformance or integration tests; Gradle/build
configuration; Vector/native routes; or unrelated modules. Those contracts either already define
the semantics consumed here or remain outside this CPU-private realization.

## Completion summary

- Completed changes: exact static first-class CPU batch-normalization inference with arbitrary
  channel axis, ordered floating promotion, typed epsilon, deterministic dual range forms, direct
  generated execution, zero resources, and schema 49.
- Files changed or created: 14 production paths, 13 test paths, and five documentation/planning
  paths; exactly four new CPU-private production types; total scope at or below 46 paths.
- Tests and validation: focused 112 passing; final uncached CPU XML 462 tests, zero failures, zero
  errors, two skips; CPU Javadoc successful; task-specific Markdown and exact repository gates
  successful; retained 281-file manifest fully verified.
- Documentation-agent review: mandatory independent clean documentation context completed without
  changing executable Java behavior; clean context/session identifier
  `01a03a19-dfac-7a61-90f5-0654784e9c74`.
- Documentation impact: CPU backend guide, glossary, this task, CPU master plan, and roadmap
  finalized; no architecture, public API, shared-contract, build, conformance, integration,
  Vector/native, or unrelated-module documentation change is required.
- Javadoc review: all 14 changed production paths inspected; affected IR, lowering geometry,
  emitter, reference, portable-family, plan, lowering-result, and prepared-executable contracts
  finalized with ownership, ranges, parameters, results, and failures.
- Glossary impact: clarified only the current CPU inference realization and schema-49 artifact
  boundary without duplicating Model training semantics.
- Unresolved issues: none for CPU 0007F1.
- Follow-up required: none for CPU 0007F1; CPU 0007F2 remains the separate Draft training frontier.

Status: Complete

## Status gate

Do not change the task, CPU master plan, or roadmap to `Complete` until every acceptance criterion,
the final uncached CPU suite, retained semantic/Class-File/member/performance evidence, final
Javadoc and documentation validation, exact scope/schema/status checks, independent clean
documentation pass, and completion summary have passed. Any missing gate leaves the task
`In progress`, `Review needed`, or `Blocked` with the exact reason.
