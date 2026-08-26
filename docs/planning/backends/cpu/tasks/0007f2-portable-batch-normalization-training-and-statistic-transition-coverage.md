# Task 0007F2: Portable Batch-Normalization Training and Statistic-Transition Coverage

## Status

Complete

## Goal

Execute exactly one fully static, resolved-layout, first-class `BATCH_NORM_TRAINING` occurrence
on the portable CPU route. Preserve Model 0021C's pure five-input/five-output meaning and
Compiler 0005B's exact output-slot roles through one direct generated, channel-owned, multi-pass
body that computes biased forward variance, correction-one running variance, explicit next
running statistics, saved batch statistics, and normalized affine output without hidden state.

## Mental model

```text
one explicit BATCH_NORM_TRAINING node
  -> validate five inputs, five distinct outputs, static channel geometry, and N >= 2
  -> assign each complete channel domain to exactly one scalar range
  -> exact mean pass -> corrected squared-deviation pass -> four channel-statistic stores
  -> normalized affine output pass
  -> publish five ordinary graph values directly; retain no state after the run
```

The range unit is one complete channel, not an input coordinate or partial reduction. Scalar and
parallel-scalar execution therefore use the same arithmetic and canonical non-channel traversal.
Parallel ranges own disjoint channels and reuse one existing exact-sum scratch slice per active
range. No partial statistic, combine phase, cross-range synchronization, or state transition is
hidden in Runtime.

## Scope

- Recognize exactly one compiled `BatchNormKind.BATCH_NORM_TRAINING` occurrence with
  `BatchNormTrainingAttrs` and ordered boundaries:

  ```text
  inputs:
    [input, scale, bias, runningMean, runningVariance]

  outputs:
    [output, nextRunningMean, nextRunningVariance,
     savedBatchMean, savedInverseStandardDeviation]
  ```

- Preserve the first-class operation identity. Never infer, reconstruct, fuse, or relabel batch
  normalization from decomposed reductions or arithmetic, and never fold inference and training
  into one CPU identity or runtime mode.
- Accept BFLOAT16, FLOAT32, and FLOAT64 independently at all five input positions. Promote in
  exact occurrence order: input with scale, then bias, running mean, and running variance. Require
  all five outputs, momentum, and epsilon to use the promoted result type.
- Revalidate Compiler 0005B's exact output descriptors and gradient-eligibility roles:

  | Slot | Shape | Required `requiresGrad` |
  |---|---|---|
  | 0 normalized output | exact input Shape | `input || scale || bias` |
  | 1 next running mean | exact `[C]` Shape | `input || runningMean` |
  | 2 next running variance | exact `[C]` Shape | `input || runningVariance` |
  | 3 saved batch mean | exact `[C]` Shape | `input` |
  | 4 saved inverse standard deviation | exact `[C]` Shape | `input` |

- Require fully static resolved descriptors, input rank at least two, normalized non-negative
  channel axis within rank, four exact rank-one channel vectors, exact input/output Shape
  equality for slot zero, exact shared channel Shape for slots one through four, non-negative
  resolved layouts, and injective writable layouts for every output.
- Derive checked geometry before any declaration or artifact work:

  ```text
  prefixCount     = product(input extents before channelAxis)
  channelCount    = input extent at channelAxis
  suffixCount     = product(input extents after channelAxis)
  reductionCount  = prefixCount * suffixCount
  inputCount      = channelCount * reductionCount
  ```

  For `channelCount > 0`, require `reductionCount >= 2`. Any product, span, address, workspace,
  or range calculation that cannot be represented by the current checked CPU contracts fails
  closed before resource declaration. Model's mathematical overflow shortcut does not authorize
  an unrepresentable CPU execution geometry.
- Use exactly one channel range form. `start/end` is a half-open channel interval. Each owned
  channel traverses every non-channel coordinate in canonical prefix-major, then suffix-major,
  logical order for every pass. No range divides a channel domain.
- Select scalar or bounded parallel-scalar execution during preparation:

  ```text
  coordinatesPerRangeItem = reductionCount
  minimumChannelsPerWorker = max(
      1,
      ceil(existingMinimumElementsPerWorker / coordinatesPerRangeItem))
  desiredRanges = min(
      usableParallelism,
      ceil(channelCount / minimumChannelsPerWorker))
  selectedRangeCount = channelCount == 0 ? 0 : max(1, desiredRanges)
  ```

  Scalar execution uses one range when work is non-empty. The range form, minimum work, and
  selected range count are cold preparation facts, not runtime tuning or generated kind dispatch.
- Treat `channelCount == 0` as completely empty work even when non-channel extents are zero or
  one. Retain truthful input/output declarations, but perform no input/vector read, scratch use,
  generated invocation, output write, square root, or worker submission.
- Use one existing alignment-eight `AGGREGATE_EXACT_STATE` workspace slice per simultaneously
  active range when `channelCount > 0`. Size one slice from the promoted result computation format
  and exact `reductionCount` through the existing exact-sum state owner. Reuse that slice
  sequentially across channels in the range. Declare no second workspace, partial/combine state,
  materialization, saved-statistic scratch, or persistent resource.
- For each channel, execute the exact finite algorithm and operation order below:

  1. Read input values in canonical non-channel order, promote them to the result computation
     format, classify NaN/infinity, and obtain the exact represented sum through the existing
     fixed-width exact-sum state.
  2. Divide the exact sum by `N` and round once to the result computation format to produce the
     internal `batchMean`.
  3. Traverse the same domain again. Form each `deviation = input - batchMean` at the result
     computation boundary. Accumulate deviations and squared deviations in ordered compensated
     binary64 sums. Form the corrected numerator exactly in this order:

     ```text
     correctionTerm = sumDeviation * sumDeviation / N
     numerator      = sumSquaredDeviation - correctionTerm
     numerator      = max(numerator, +0) only when the finite residue is negative
     biasedVariance   = numerator / N
     unbiasedVariance = numerator / (N - 1)
     ```

     Round biased and unbiased variance separately to the result computation format. Never derive
     one from the other, use `E[x*x] - mean*mean`, clamp a legitimate input/state value, or use
     correction zero for the running transition.
  4. Compute saved inverse standard deviation in this exact typed order:

     ```text
     radicand = biasedVariance + epsilon
     root     = sqrt(radicand)
     savedInv = 1 / root
     ```

  5. Compute the transitions with momentum as the new-batch weight. Round every shown
     subtraction, multiplication, and addition at the result computation boundary:

     ```text
     oneMinusMomentum   = 1 - momentum
     nextRunningMean    = oneMinusMomentum * runningMean
                          + momentum * batchMean
     nextRunningVariance = oneMinusMomentum * runningVariance
                           + momentum * unbiasedVariance
     ```

  6. Store slots one through four once for the channel in this order: next running mean, next
     running variance, saved batch mean, saved inverse standard deviation.
  7. Traverse the non-channel domain a third time and compute each slot-zero coordinate exactly:

     ```text
     centered     = input - batchMean
     standardized = centered * savedInv
     scaled       = standardized * scale
     output       = scaled + bias
     ```

     Do not replace multiplication by saved inverse standard deviation with division, fuse
     multiply-add, precompute `alpha/beta`, or reassociate the affine expression.
- Use FLOAT32 arithmetic boundaries for BFLOAT16/FLOAT32 results and FLOAT64 boundaries for
  FLOAT64 results. The exact sum and compensated binary64 variance state are the explicitly
  selected equal-or-wider reduction intermediates. BFLOAT16 encodes only at each final output
  store; FLOAT32/FLOAT64 use matching primitive stores. Generated and identical-algorithm direct
  Java must be raw-bit equal.
- Preserve Model 0021C's special-value classes:
  - any input NaN or infinity in a non-empty channel produces CPU canonical NaN for its mean,
    both variance estimates, saved inverse standard deviation, normalized output coordinates,
    and both next-statistic formulas after ordinary arithmetic;
  - a finite constant domain, including signed zeros, produces positive-zero biased/unbiased
    variances; mean zero sign follows the exact selected sum/division rule;
  - old running variance is neither read for forward normalization nor validated, repaired, or
    clamped; it affects only slot two;
  - affine and old-statistic NaN/infinity affect only their written formula roles;
  - momentum endpoints use the written formula, so zero coefficients do not suppress NaN or
    infinity;
  - finite overflow, underflow, signed zero, root, reciprocal, and affine behavior follow the
    frozen typed order above.
- Deduplicate repeated logical input `ValueId` values in first-occurrence order and retain an
  immutable five-position-to-boundary map. Input/input alias is legal read sharing. Require all
  five output `ValueId` values to be distinct from every input and from one another.
- Declare each unique input once followed by all five outputs in semantic output order. The
  generated entry receives unique input carriers, five output carriers, one scratch segment,
  packed primitive geometry, and primitive `long start, long end`.
- Before mutation or worker submission, validate all ten semantic positions, deduplicated carrier
  compatibility, liveness and worker access, native order, alignment, complete spans, every
  output's writability/injectivity, the exact workspace, every workspace/buffer overlap, every
  output/input overlap, and every output/output overlap. Validation failure writes nothing and
  submits no work.
- After successful validation, store directly into all five ordinary output representations.
  There is no shadow result, commit phase, rollback, in-place running-statistic update, hidden
  cross-run state, or special saved-output lifetime. Arithmetic is non-throwing; an unrelated
  later generated/worker failure follows the existing Runtime contract and may expose already
  written output coordinates.
- Support every unique boundary's exact `short[]`, `float[]`, `double[]`, or native-order
  `MemorySegment` carrier according to its own type, including all-heap, all-segment, and mixed
  patterns, arbitrary legal offsets, positive or zero input strides, and arbitrary injective
  output strides.
- Add one focused training IR, one arbitrary-axis training lowerer/geometry owner, one direct
  emitter, and one independent reference owner. Reuse current exact-sum, carrier, access, route,
  preparation, finalization, worker, overlap, artifact, and multi-output seams. Do not extend the
  inference IR into a training/inference union or create a generic normalization interpreter.
- Emit one deterministic final, field-free, constructor-free generated class per specialization
  with exactly one public static typed `invoke` entry and no helper method.
- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly once from 49 to 50. The family, five-output
  signature, momentum/epsilon bits, range/pass/arithmetic identity, exact-state shape, boundary
  map, and generated body change compatibility. Schema 49 and earlier envelopes are safe misses;
  no migration reader or historical evidence mutation is added.
- Use a frozen optimal clean Java implementation with identical validation, channel ownership,
  exact-sum state, pass/dataflow shape, addressing, typed arithmetic, stores, and workspace reuse
  as the design, review, and performance oracle. Generated hot work may call only required JDK
  primitive/raw-bit, `Math.sqrt`, and typed `MemorySegment` operations and must call no Synaptik
  numerical, validation, or reference helper.

## Out of scope

- `BATCH_NORM_INFERENCE` semantic or implementation changes; Layer/RMS/softmax/reduction changes;
  a shared normalization identity; decomposed recognition; inference/training mode flags; or
  folding inference and training artifacts together.
- Stateful layers, parameters, buffers, assignment of next statistics, cross-step transport,
  training/evaluation mode, optimizer/session/checkpoint ownership, mutation, serialization, or
  hidden state.
- Compiler inference, deferred constraints, autograd formulas, gradient eligibility, saved-value
  capture/liveness, publication planning, or optimization changes. Slots three and four remain
  existing same-occurrence compiler auxiliaries, not backend-created tape values.
- Optional operands, default values, implicit constants, scalar/full-rank channel operands,
  optional momentum, batch counters, cumulative average, alternate momentum conventions,
  configurable correction, variance repair, or Tensor-valued scalars.
- Dynamic or symbolic Shapes, unresolved layouts, runtime channel axes, rank below two, positive
  channels with reduction count below two, negative strides, non-injective outputs, or
  unrepresentable counts/spans/addresses/resources.
- FLOAT16, integral, BOOL, complex, quantized, sparse, or unsigned arithmetic; configurable
  accumulation/result type; public casts; relaxed math; or cross-backend bitwise equality.
- Partial channel reductions, reduction/combine trees, non-channel range splitting, atomic
  statistics, per-coordinate task submission, a denominator/statistic table, or more than one
  workspace.
- Vector API, native/vendor routes, fusion with adjacent work, materialization, autotuning,
  measured candidate selection, dynamic dispatch, or scalar-reference Runtime fallback.
- Generated allocation, boxing, reflection, method handles, `invokedynamic`, dynamic constants,
  bootstrap methods, `Object` descriptors, collection/map/string lookup, or avoidable semantic,
  carrier, layout, output-role, pass, or kind dispatch in hot work.
- Public/shared API, Model, Compiler, Training, Runtime, Prepare, backend-contract, Trace,
  Planning, Engine, dependency, module-boundary, Gradle/toolchain, architecture, conformance, or
  integration changes beyond the CPU-private generator schema.
- Reopening historical CPU 0007A1D or creating a detailed CPU 0008 specification.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007F Layer/RMS normalization`](0007f-portable-layer-and-rms-normalization-coverage.md)
- [`CPU 0007F1 batch-normalization inference`](0007f1-portable-batch-normalization-inference-coverage.md)
- [`Model 0021C batch-normalization training`](../../../modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md)
- [`Compiler 0005B normalization inference and gradients`](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [`Prepare 0001 resource declaration`](../../../modules/prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md)
- [`Prepare 0002 finalization handoff`](../../../modules/prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Model owns the first-class kind, exact ordered roles, arbitrary logical channel axis, Shapes,
  reduction domain, biased/unbiased distinction, momentum convention, epsilon, promotion,
  computation format, metadata, and special-value classes. Compiler owns first-class capture,
  exact five-output inference/validation, output-slot-aware gradients, and saved-value lifetime.
  CPU owns only truthful static admission, lowering, numerical realization, resources, access
  validation, and generated execution.
- CPU must analyze only the explicit `BATCH_NORM_TRAINING` node. Equivalent decomposed graphs
  remain ordinary nodes; `BATCH_NORM_INFERENCE` retains its separate schema-49 identity.
- The completed Model and Compiler contracts are implementation inputs, not files to revise. Any
  disagreement in signature, output order, descriptor roles, domain validity, promotion, scalar
  type, formula, or saved-value meaning is a stop condition.
- CPU analysis chooses the route and declares every buffer/workspace requirement before shared
  assignment. CPU finalization may construct or reuse the schema-50 artifact only afterward and
  may not change the route or resource set. Runtime receives one immutable prepared executable.
- Planning continues to select CPU ownership only. Shared Prepare assigns existing buffer and
  workspace slots without interpreting channels, statistics, passes, momentum, or saved outputs.
- Generated code must preserve the optimal clean Java specialized algorithm, hot-loop/dataflow
  shape, operation order, workspace reuse, and avoidable-overhead profile. A reference call,
  generic bridge, decomposed pointwise/reduction sequence, or runtime output-role/pass switch is
  not a specialized implementation.
- Any required public/shared/build/architecture/conformance/integration change, new resource
  kind, second workspace, partial/combine representation, unresolved semantic choice, or path/type
  ceiling violation is a stop/replan condition.

## Current-code evidence

- Model 0021C and current `TensorBatchNormTrainingExpressions` fix the five ordered inputs, five
  ordered outputs, exact Shapes, ordered promotion, typed momentum/epsilon, `C == 0 || N >= 2`,
  metadata, saved-statistic meanings, formulas, and pure transition boundary.
- Current `ReductionNormalizationInference` revalidates all five descriptors and deferred domain
  obligation. `NormalizationGradientRules` retrieves exact same-occurrence slots three/four and
  assigns public-output gradients by exact output slot. Neither decomposes the forward node.
- `CpuCapabilityProvider` and `CpuPartitionLowering` admit only batch inference today; training
  therefore fails closed.
- Schema-49 `CpuBatchNormInferenceIr`, lowering, emitter, reference, preparation, and execution
  prove arbitrary-axis geometry, ordered mixed floating carriers, first-occurrence input
  deduplication, static layout checks, direct generated invocation, and cold range selection. Its
  zero-workspace coordinatewise algorithm is not the training algorithm and remains separate.
- `CpuStatisticalReductionEmitter`, `CpuExactSumEmitter`, advanced-reduction geometry, and Layer
  normalization already prove exact-sum per-range scratch, corrected squared-deviation arithmetic,
  workspace slicing, and direct multi-pass generated bodies. Training reuses these narrow seams
  but owns one separate five-output emitter and geometry.
- Ordering and dropout already prove direct multi-output boundary declaration and invocation.
  Existing generic overlap logic handles at most their current shapes; training must explicitly
  validate all five output/input and ten output/output pairs rather than inheriting a one/two-
  output assumption.
- Existing preparation has one optional CPU workspace and `AGGREGATE_EXACT_STATE`; no shared
  resource kind or contract change is needed for one exact-state slice per selected range.

## Package impact

Existing CPU-private packages changed:

- `io.github.pho001.synaptik.backend.cpu` — exact truthful capability admission.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — immutable training identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — arbitrary-axis channel-domain
  geometry, boundary mapping, and exact-state sizing.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct three-pass/five-output
  generated body.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent differential oracle.
- Existing `route.portable`, `prepare`, `executable`, and `cache` packages — scalar realization,
  declaration, workspace assignment, complete cold validation, invocation, and schema-50
  compatibility.

Packages added: None.

Type placement:

- Add `CpuBatchNormTrainingIr` in `internal.ir` for exact five-position input types, result type,
  raw momentum/epsilon bits, rank/channel axis, algorithm/pass/arithmetic version, exact-state
  shape, position map, and ten ordered access plans.
- Add `CpuBatchNormTrainingLowering` in `internal.lowering` for static
  prefix/channel/suffix/reduction geometry, five output layouts, unique input boundaries, exact
  state size, canonical traversal, and packed cold geometry.
- Add `CpuBatchNormTrainingEmitter` in `internal.codegen.emit` for the sole direct exact-mean,
  corrected-variance, transition, saved-statistic, and normalized-output body.
- Add `CpuBatchNormTrainingReferenceKernel` in `internal.reference`; it independently derives
  channel domains and expected five-output values without production lowering, emission,
  binding, or generated helpers.
- Add no second emitter, validator type, state owner, generic normalization facade, manager,
  registry, or training/inference union.

## Affected files

Expected production/Javadoc paths:

- `CpuCapabilityProvider.java` and CPU package Javadoc;
- `internal/ir/CpuPortableKernelIr.java`, new `CpuBatchNormTrainingIr.java`, and IR package Javadoc;
- `internal/lowering/CpuPartitionLowering.java`, new `CpuBatchNormTrainingLowering.java`, and
  lowering package Javadoc;
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new
  `CpuBatchNormTrainingEmitter.java`, existing exact-sum/carrier seams only where required, and
  emitter package Javadoc;
- `internal/cache/CpuGeneratorSchema.java`, `CpuKernelSpecialization.java` only if exact scratch or
  entry validation requires it, and cache package Javadoc;
- `internal/route/portable/CpuPortableRoutePlan.java` and package Javadoc;
- `internal/prepare/CpuPartitionPreparationPlan.java`, `CpuPartitionPreparer.java`,
  `CpuPartitionFinalizer.java`, and package Javadoc;
- `internal/executable/CpuPreparedExecutable.java` and executable package Javadoc; and
- new `internal/reference/CpuBatchNormTrainingReferenceKernel.java`, the minimal existing
  differential seam if required, and reference package Javadoc.

Expected test paths:

- capability and internal-package inventory tests;
- new `CpuBatchNormTrainingIrTest`, `CpuBatchNormTrainingLoweringTest`,
  `CpuBatchNormTrainingGeneratedKernelTest`, `CpuBatchNormTrainingEvidenceTest`, and
  `CpuBatchNormTrainingReferenceTest`;
- generated-class shape/member, specialization/cache, route, prepare/finalization, executable,
  exact-state workspace, and reference-differential tests only where exact inventories or behavior
  change; and
- the historical pointwise-ledger test only for its current-schema assertion; retained historical
  evidence resources remain immutable.

Expected documentation/planning paths in the implementation change:

- `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  global roadmap. Tensor/Compile/Training APIs, Model/Compiler plans, architecture, shared Prepare,
  backend-contract, conformance, integration, and build files are review-only unless a stop
  condition is reached.

## Maximum scope

The complete implementation, tests, Javadocs, and documentation may modify or create at most 48
repository paths: at most 28 production/Javadoc paths, 15 test paths, and exactly the five named
documentation/planning paths. Exactly four new CPU-private production types are permitted: the
IR, lowerer, emitter, and independent reference owner listed above.

This exceeds the planning guide's ordinary guardrail because one truthful five-output generated
family must update the current sealed IR, lowering, generator, schema/cache, route, one-workspace
preparation, finalization, executable binding, package-Javadoc, inventory, and differential seams
atomically. The behavioral core remains four focused types in one module. Splitting declaration,
multi-output binding, or execution would temporarily advertise a family that cannot publish all
of its semantic outputs and would duplicate the same module/evidence gate.

A 49th path, fifth new production type, second emitter/validator, new resource kind, second
workspace, public/shared/build/architecture/conformance/integration path, inference semantic
change, partial/combine route, or CPU 0008 behavior is a stop/replan condition. An omitted expected
path does not authorize an unrelated path.

## Detailed generated-code design boundary

- `CpuBatchNormTrainingIr` is instruction-free. Its cache identity includes family, five semantic
  input types, result type, raw momentum/epsilon bits, rank/channel axis, algorithm/arithmetic/pass
  version, exact-state limb/slice shape, five-position map, ordered unique-input/five-output access
  plans, and scratch-bearing entry shape.
- `CpuBatchNormTrainingLowering.Geometry` retains concrete static extents/layouts,
  prefix/channel/suffix/reduction/input counts, output positions, canonical traversal, and exact
  scratch size. Concrete compatible extents, strides, offsets, and selected range count stay cold
  unless they change exact-state width or emitted bytes.
- The generated outer loop owns channels. Each channel performs exact-sum pass, corrected-variance
  pass, four scalar statistic stores, and normalized-output pass. Dense arrays may use cold-proved
  integer indexes; arbitrary layouts use primitive odometers/increments. No hot division/modulo
  or full coordinate reconstruction is repeated per element when a direct increment is provable.
- One specialization has one generated artifact. The existing four-candidate/one-artifact/
  zero-unroll-variant budget remains unchanged. Training is never materialized or Vector-selected.
- Schema 50 is current-only. Schema-49 and earlier envelopes safely miss and regenerate; no
  compatibility alias, migration reader, or retained-evidence rewrite is added.

## Acceptance criteria

- Capability admits exactly first-class `BATCH_NORM_TRAINING` with `BatchNormTrainingAttrs`, five
  inputs, five outputs, all current BFLOAT16/FLOAT32/FLOAT64 promotion combinations, exact typed
  momentum/epsilon, static Shapes/layouts, `C == 0 || N >= 2`, and exact output descriptors.
- `BATCH_NORM_INFERENCE` remains separately admitted and unchanged. Source-backed tests prove an
  equivalent decomposed graph remains ordinary operations and never becomes either batch IR.
- Lowering records exact types/scalars/rank/axis/counts, five-position input map, unique boundaries,
  five distinct outputs, access plans, exact-state shape, and channel range identity. Checked
  geometry and resource validation complete before declarations.
- Semantic/differential coverage includes channel axis first/middle/last; rank two and higher;
  channel extent zero/one/larger; reduction count two and larger; rejected positive-channel
  counts zero/one; same/mixed types; dense/general/zero-stride inputs; arbitrary injective output
  layouts; offsets; repeated inputs; all heap/all segment/mixed carriers; and mixed output Shapes.
- Formula tests lock exact sum/divide mean, corrected squared-deviation numerator, separately
  divided biased/unbiased variances, epsilon placement, reciprocal square root, new-batch-weight
  transitions, statistic store roles/order, saved values, and multiply-based normalized affine
  order with the exact FLOAT32/FLOAT64 boundaries.
- The Model `[1,2,3]` example with scale `2`, bias `0.5`, old mean `10`, old variance `4`, momentum
  `0.25`, and epsilon `1e-5` produces mean `2`, biased variance `2/3`, unbiased variance `1`, next
  mean `8`, next variance `3.25`, saved inverse standard deviation
  `1/sqrt(2/3+1e-5)`, and normalized output approximately
  `[-1.949471, 0.5, 2.949471]` under the selected typed tolerance.
- Special-value tests cover input/affine/old-statistic NaNs and infinities, both signed zeros,
  finite constants, negative old variance, momentum `-0`, `+0`, and `1`, subnormals,
  overflow/underflow, positive-zero variance, saved inverse standard deviation, and unsuppressed
  zero-coefficient NaN/infinity.
- An independent StrictMath/high-precision oracle checks ordinary finite results within 16 result
  ulps for FLOAT64, four result ulps for FLOAT32, and one represented BFLOAT16 ulp. These are the
  existing conservative maxima already justified by CPU 0007D's corrected variance and CPU
  0007F's normalization evidence, the two numerical components composed here; this task does not
  invent a looser family-specific threshold. Exact special classes, infinities, and zero signs use
  class/raw-sign assertions. Generated and frozen identical-algorithm direct Java are raw-bit
  equal for all five outputs.
- Resource declarations contain unique inputs in first-use order, then five outputs in semantic
  order, plus exactly one alignment-eight exact-state workspace when work is non-empty. They
  contain no materialization, second workspace, partial/combine value, hidden saved buffer, or
  persistent state.
- Empty channel work performs no reads, scratch use, square root, generated call, store, or worker
  submission and preserves every carrier and canary.
- Cold binding validates the complete carrier/span/layout/workspace/overlap matrix before mutation
  or submission. All input/input aliases remain legal read sharing; all output/input and
  output/output overlaps fail without writes or submission.
- Scalar and parallel-scalar results are raw-bit deterministic across complete-channel ranges,
  worker counts, repeated runs, and concurrent runs. Each non-empty channel is owned once; each
  statistic coordinate is written once; each normalized coordinate is written once; every active
  range receives a disjoint exact-state slice.
- Generated classes are deterministic, final, field-free, constructor-free, and expose exactly
  one typed static `invoke` with unique input carriers, five output carriers, scratch, geometry,
  and `long start/end`. There is no `Object`, bridge descriptor, kind/pass/output-role argument,
  validation/reference call, or helper method.
- Complete `javap -c -p` and `javap -v -p` inspection proves exact pass and store order, direct
  typed carrier/address work, generation-time family/type/axis/carrier specialization, and absence
  of allocation, boxing, reflection, method handles, `invokedynamic`, dynamic constants,
  bootstrap methods, collections/maps/strings, semantic dispatch, and avoidable address work.
- A complete member allowlist permits only required primitive/raw-bit operations, `Math.sqrt`, and
  typed `MemorySegment` access. Any unexpected member is a failure; generated classes reference no
  Synaptik numerical, validation, exact-sum runtime, or reference helper.
- Specialization/cache identity distinguishes every byte-shaping fact named above. Compatible
  cold bases, offsets, ranges, slots, workers, graph/run identities, and values do not change
  identity. Different raw momentum or epsilon bits are incompatible.
- Schema advances exactly 49 to 50. Older envelopes safely miss; completed task records and
  retained evidence stay immutable and historical.
- Freeze exactly eight training targets. `S` means a matching native-order `MemorySegment`.
  Vector layout pairs are `(offset,stride)`. `BNT-F64-A1` uses usable parallelism one to measure
  the scalar range form; the other seven use usable parallelism four. Every target uses
  `minimumElementsPerWorker = 4096`; parallel targets produce four channel ranges unless the exact
  channel count makes fewer ranges truthful.

  | Target | Shape / axis | Inputs -> result | Carriers | Layout |
  |---|---|---|---|---|
  | `BNT-BF16-A1` | `[32,64,256]` / 1 | all BFLOAT16 -> BFLOAT16 | all `short[]` | dense |
  | `BNT-F32-A1` | `[32,64,256]` / 1 | all FLOAT32 -> FLOAT32 | all `float[]` | dense |
  | `BNT-F64-A1` | `[32,64,256]` / 1 | all FLOAT64 -> FLOAT64 | all `double[]` | dense |
  | `BNT-F32-A0` | `[128,4096]` / 0 | all FLOAT32 -> FLOAT32 | all `float[]` | dense |
  | `BNT-F32-A2` | `[32,256,64]` / 2 | all FLOAT32 -> FLOAT32 | all `float[]` | dense |
  | `BNT-MIX-F64` | `[16,32,64]` / 1 | FLOAT32/FLOAT64/BFLOAT16/FLOAT32/FLOAT64 -> FLOAT64 | mixed arrays/`S` -> five mixed FLOAT64 outputs | frozen legal general layouts |
  | `BNT-MIX-F32` | `[16,32,64]` / 1 | BFLOAT16/FLOAT32/BFLOAT16/FLOAT32/BFLOAT16 -> FLOAT32 | mixed arrays/`S` -> five mixed FLOAT32 outputs | frozen legal general layouts |
  | `BNT-REPEAT` | `[32,64,128]` / 1 | all FLOAT32 -> FLOAT32 | repeated exact vector input boundary; mixed outputs | dense input, legal strided vectors/statistics |

  Freeze unchanged FLOAT32 batch inference, Layer Norm, `VARIANCE`, and pointwise `ADD` controls
  at their retained CPU-0007F1/0007F/0007D/0005 definitions. This matrix is proportional to the
  new family: all result formats, three channel-axis positions, dense/general/mixed carriers,
  repeated inputs, five-output stores, exact-state scratch, and both scalar/parallel code paths
  are represented without claiming universal layout or workload performance.
- Every target and control passes generated/direct `<= 1.15x` in each of five accepted isolated
  forks and for the median of fork medians. The frozen optimal clean Java oracle performs
  identical useful validation, arithmetic, passes, addressing, conversions, workspace reuse, and
  stores and is never slowed, padded, or routed through production helpers. Reject and retain a
  whole sample when any semantic, checksum, environment, target, or control gate fails.
- Focused tests, one final uncached CPU suite, Javadoc, retained semantic/Class-File/decompilation/
  member/operation-count/checksum/five-fork evidence, exact scope, schema/status/staging, and
  whitespace gates pass. A distinct clean documentation context finalizes affected Javadocs, CPU
  guide, glossary impact, task, master plan, and roadmap without repeating stable Java/performance
  work.

## Tests / validation

The implementation context runs focused tests after stabilization, then one final CPU suite:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormTrainingIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuBatchNormTrainingGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuBatchNormTrainingReferenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest
./gradlew :backends:cpu:test --rerun-tasks
```

Retain one immutable evidence bundle outside the repository containing exact source/probe inputs,
environment/JVM facts, all eight targets and four controls, specializations, complete
`javap -c -p` and `javap -v -p`, member and operation-count reports, semantic/accuracy/range/
workspace/alias/canary results, five isolated `-Xms1g -Xmx1g` fork outputs, rejected whole
samples, summaries, inventory, and a SHA-256 manifest. Each fork uses at least five randomized
warmup batches, nine randomized measured rounds, adaptive batches of at least 25 ms, deterministic
inputs, randomized generated/direct order, and raw/checksum verification.

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
exact path/type ceilings, schema 50 and historical-evidence wording, CPU 0007A1D Review needed,
CPU 0007F/0007F1 Complete, CPU 0007F2 Complete only after every gate, CPU 0008 and later tasks
Draft without detailed specifications, and empty staging.

Repository-wide, architecture, backend-conformance, and integration suites remain deferred to CPU
0009 or continuous integration because the implementation changes one concrete backend without a
shared boundary. A stop condition that changes a shared boundary requires replanning and
proportionate validation.

## Dependencies

- Complete Model 0021C owns the exact first-class five-input/five-output meaning, arbitrary channel
  axis, Shapes, promotion, typed scalars, domain obligation, biased/unbiased statistics, saved
  values, transition, metadata, and special-value contract.
- Complete Compiler 0005B and current inference/preflight/gradient code own exact five-output
  capture and validation, output-slot roles, same-occurrence saved slots three/four, and
  first-class identity. CPU changes no formula or gradient behavior.
- Complete CPU 0007F supplies exact-sum per-range workspace, mixed-floating normalization
  carriers, multi-pass generated evidence, and workspace slicing. Complete CPU 0007F1 supplies
  arbitrary-axis batch geometry, five-position input deduplication, static admission, direct
  generated batch execution, and schema-49 compatibility. Neither existing IR is reused as the
  training semantic identity.
- Existing ordering/dropout multi-output paths and shared Prepare/Runtime contracts carry the
  five ordinary output boundaries and one workspace unchanged. CPU 0007A1D remains historical
  Review needed and is not a dependency.

All dependencies and semantic/resource decisions are settled. No architecture or shared-contract
blocker is known within the fail-closed static complete-channel scope.

## Follow-up tasks

- CPU 0008 remains Draft for the remaining heavy portable families and keeps its corrected
  Model/Compiler prerequisites. No detailed 0008 specification is created here.
- CPU 0008C must preserve the prohibition on recognizing decomposed normalization as a first-class
  semantic kernel.
- CPU 0009 or continuous integration owns repository-wide portable coverage, conformance, and
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
You are the clean implementation agent for Synaptik CPU task 0007F2. Work on the existing
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, CPU task 0007F2
and its dependencies, Model 0021C and its implemented source/tests/Javadocs, Compiler 0005B and
current normalization inference/preflight/gradient code, CPU 0007F/0007F1, and current CPU
capability/IR/lowering/exact-sum/emitter/reference/route/prepare/finalization/executable/cache
source and tests plus documentation rules and applicable profiles.

Implement exactly CPU 0007F2 within its 48-path/four-new-type ceilings. Admit only explicit
BATCH_NORM_TRAINING with five ordered inputs and outputs, complete-channel scalar/parallel-scalar
ranges, exact mean state, corrected biased/unbiased variances, typed momentum/epsilon, explicit
transitions, saved statistics, direct five-output stores, exact overlap validation, one existing
workspace, and schema 49-to-50 identity. Preserve validation-before-mutation, first-occurrence
input deduplication, separate inference identity, and every fail-closed exclusion. Generated
bytecode must match the frozen optimal clean Java algorithm, hot-loop/dataflow/pass shape,
operation boundaries, resource reuse, and avoidable-overhead profile. Satisfy every semantic,
accuracy, resource, alias, Class-File/member/operation-count, checksum, control, and five-fork
performance gate without weakening or slowing the oracle.

Stop on any ceiling, architecture uncertainty, semantic conflict, shared resource change, second
workspace, partial/combine requirement, inference change, or scope conflict. Hand the stabilized
diff, final CPU XML, and retained evidence to a distinct clean documentation-focused context to
finalize Javadocs, CPU guide, glossary, and planning in the same overall change. Do not mark
Complete until all gates pass. Do not commit, push, or stage.
```

## Documentation-focused finalization requirement

After executable Java and tests stabilize, a distinct clean documentation-focused agent/thread
must inspect the actual diff and retained evidence. It must apply General, API/Javadoc, Backend
Guide, Planning, and Example profiles as relevant; finalize all affected Javadocs, the CPU guide,
glossary impact, this task, master plan, and roadmap; inspect rendered Javadocs; reuse successful
Java/performance evidence unless executable behavior changes; and record reasoned no-change
conclusions for Tensor/Compile/Training APIs, Model/Compiler contracts, architecture/ADRs/tests,
shared Prepare/backend-contract/Runtime, conformance/integration, Gradle, other modules, and the
accepted CPU 0008 ordering correction.

## Local decisions

- Use a separate training IR/lowerer/emitter/reference quartet. Training's reductions, exact-state
  workspace, five outputs, and statistic transitions are not inference identity or trailing
  Layer/RMS geometry.
- Own complete channels in scalar and parallel-scalar ranges. This keeps one deterministic
  reduction order and eliminates partial/combine state. Small channel counts may underuse workers;
  that limitation is safer than introducing unplanned synchronization or workspace topology.
- Reuse one exact-sum scratch slice per active channel range. The saved outputs are ordinary graph
  buffers, not workspace, persistent prepared state, or a runtime tape.
- Use exact sum for the mean and the completed corrected two-pass variance precedent. Compute
  biased and unbiased results by separate division from one corrected numerator. This avoids the
  unstable raw-moment formula and does not derive one estimate from the other.
- Preserve multiplication by saved inverse standard deviation and explicit multiply-then-add
  affine order. Preserve written momentum arithmetic even at zero/one endpoints.
- Keep compatible concrete extents/layouts cold except where exact-state width changes generated
  bytes. Bake family, types, raw scalars, axis/rank, algorithm/pass/state shape, boundary map,
  access/carriers, and entry shape.
- Advance compatibility exactly from schema 49 to 50. Older envelopes are safe misses and require
  no migration reader.

## Known limitations

- Only fully static resolved-layout scalar/parallel-scalar forward execution is planned. Dynamic
  Shapes, split reductions, Vector/native routes, fusion, materialization, and hidden state remain
  unsupported.
- Parallelism is bounded by channel count because each reduction domain has one owner. The task
  makes no claim that channel-poor workloads saturate the CPU.
- Finite results follow the selected CPU algorithm and tolerance matrix. Raw-bit identity is
  required only between generated and identical-algorithm direct Java, not across backends.
- The performance gate covers only the frozen proportional matrix and recorded environment; it is
  not a general throughput, tuning, or layout-optimality claim.

## Validation evidence

- Implementation context `01a03a7a-07e6-7473-a5fd-5f263df4236d` and resumed performance context
  `01a03c47-6f66-7372-84bf-b8937d492186` supplied the stabilized executable evidence. The final
  focused matrix passed 10 suites/114 tests with zero failures, errors, or skips. The final
  uncached `./gradlew :backends:cpu:test --rerun-tasks` passed 86 suites/472 tests with zero
  failures or errors and two expected skips. Documentation-only Java changes followed, so clean
  documentation context `01a03c62-6576-7df0-bb6b-5f454918fb72` did not repeat either Java suite.
- The retained bundle contains eight deterministic schema-50 generated classes plus complete
  `javap -c -p` and `javap -v -p`, member allowlists, forbidden-reference scans, and operation
  counts. All eight classes use direct odometer traversals, one `Math.sqrt` site, and four semantic
  division sites, with no allocation, boxing, reflection, method handle, `invokedynamic`, dynamic
  constant, collection/map/string dispatch, invocation-time layout construction, or Synaptik
  numerical/reference helper.
- Five accepted isolated Java 26.0.1 `-Xms1g -Xmx1g` forks passed all 60 per-fork and 12
  aggregate generated/optimal-clean-Java ratios at `<= 1.15x`. The worst fork ratio was
  `1.123421082x` and the worst aggregate ratio was `1.110678870x`, both `BNT-REPEAT`. One whole
  environment/classpath sample was rejected before measurement and retained; no measured ratio
  sample was discarded.
- The read-only retained evidence root is
  `/private/tmp/synaptik-cpu-0007f2-retained-evidence-20260825`. Both `shasum -a 256 -c
  SHA256SUMS` and `shasum -a 256 -c SHA256SUMS.sha256` passed. The manifest digest is
  `d1138b75924cea2b1bbce6ba127213eb2181de7156884b857ce1e475b9b95edb`.
- Clean documentation context `01a03c62-6576-7df0-bb6b-5f454918fb72` applied the General,
  API/Javadoc, Backend Guide, and Planning profiles. It finalized every new training-type
  contract, the changed schema/plan/executable contracts, the CPU guide, glossary, this task, CPU
  master plan, and roadmap. `./gradlew :backends:cpu:javadoc` passed after final Javadoc edits;
  generated pages for the four new types and changed constructors/records were inspected. Local
  Markdown links and anchors, fences, final newlines, whitespace, status, schema, path/type/test,
  staging, and retained-manifest checks passed. `git diff --check` and
  `git diff --cached --check` passed; staging remained empty.
- One final sandboxed Javadoc invocation was denied access to Gradle's user-home wrapper lock
  before Gradle started. The immediate approved `./gradlew :backends:cpu:javadoc` rerun passed
  with only the two expected incubating Vector API warnings; this was an environment-access retry,
  not a source or test failure.
- Final dirty-worktree count is 47 paths: 25 production/Javadoc, 15 test, five named 0007F2
  documentation/planning paths, and the two preserved unrelated NN/Engine planning paths. The
  bounded 0007F2 scope is therefore 45 paths, within 48 overall, 28 production/Javadoc, and 15
  test ceilings, with exactly four new CPU-private production types and exactly the five required
  task documentation paths.
- Status checks confirm CPU 0007A1D remains `Review needed`; CPU 0007F, 0007F1, and 0007F2 are
  `Complete`; CPU 0008 and later rows remain `Draft` without detailed specifications; and the
  Conv1d/Conv3d-before-general-DAG ordering correction remains intact.

## Implementation notes

- Added one separate training IR, lowerer/geometry owner, direct emitter, and independent
  clean-Java reference owner. Existing inference identity and semantics remain unchanged.
- Capability, lowering, preparation, finalization, and executable binding now carry exactly one
  explicit five-input/five-output static occurrence. Repeated inputs share first-occurrence read
  boundaries; five outputs remain distinct and are validated against every input, output, and the
  exact workspace before mutation or worker submission.
- Complete-channel scalar and parallel-scalar ranges reuse one exact-sum state slice per active
  range. The generated body preserves the exact mean, corrected squared-deviation numerator,
  separate biased/unbiased divisions, biased saved inverse standard deviation, new-batch-weight
  momentum transitions, four statistic stores, and multiply-based normalized affine output.
- Schema advanced exactly once from 49 to 50. Earlier envelopes remain safe misses and retained
  historical evidence was not modified.
- No architecture or ADR update is needed: the implementation stays inside the concrete CPU
  backend's existing analysis/finalization/runtime boundary. No public Tensor, Compile, Training,
  Engine, Runtime, Prepare, backend-contract, Trace, Planning, conformance, integration, Gradle,
  dependency, or module-boundary contract changed. Existing Model 0021C and Compiler 0005B
  contracts remain accurate. Nine affected CPU package Javadocs now include the training family
  and current schema while preserving their package purpose, visibility boundary, and unsupported
  internal-surface warning. CPU package inventory tests cover the four new private owners.

## Completion summary

- Completed changes: implemented and documented portable schema-50 first-class batch-normalization
  training with explicit five-input/five-output semantics, complete-channel deterministic ranges,
  exact-state workspace, saved statistics, and pure running-statistic transitions.
- Files changed or created: 25 production/Javadoc paths, 15 test paths, and five 0007F2
  documentation/planning paths; two unrelated pre-existing NN/Engine planning edits were preserved
  unchanged.
- Tests and validation: focused 10 suites/114 tests and final CPU 86 suites/472 tests passed;
  eight-class semantic/Class-File/member/operation-count evidence and all five-fork performance
  gates passed; final CPU Javadoc, Markdown, manifest, scope, status, staging, and whitespace gates
  passed.
- Documentation-agent review: clean context `01a03c62-6576-7df0-bb6b-5f454918fb72` finalized
  affected Javadocs, CPU guide, glossary, task, master plan, and roadmap without changing
  executable behavior or repeating stable Java/performance work.
- Documentation impact: the CPU guide now explains the executable training lifecycle and bounded
  evidence; planning records identify 0007F2 as Complete and CPU 0008 as the next Draft frontier.
- Javadoc review: all four new types, nine affected packages, and changed
  record/constructor/schema contracts describe semantics, parameters, results, ownership,
  mutation, resources, and failures where applicable.
- Glossary impact: batch-normalization training and generated-artifact entries now distinguish
  biased saved statistics, unbiased running transition, complete-channel exact-state execution,
  and schema 50 from inference and hidden state.
- Unresolved issues: None.
- Follow-up required: None. CPU 0008 remains separate Draft work.

Status: Complete

## Status gate

Do not change the task, CPU master plan, or roadmap to `Complete` until every acceptance criterion,
the final uncached CPU suite, retained semantic/Class-File/member/operation-count/performance
evidence, final Javadoc and documentation validation, exact scope/schema/status checks, independent
clean documentation pass, and completion summary have passed. Any missing gate leaves the task
`In progress`, `Review needed`, `Blocked`, or `Incomplete` with the exact reason.
