# Task 0007F2: Portable [Batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[Normalization](../../../../glossary.md#term-normalization) Training and [Statistic](../../../../glossary.md#term-statistic)-Transition Coverage

## Status

Complete

## Goal

Execute exactly one fully static, resolved-[layout](../../../../glossary.md#layout), first-class `BATCH_NORM_TRAINING` [occurrence](../../../../glossary.md#term-occurrence)
on the portable [CPU](../../../../glossary.md#term-cpu) route. Preserve [Model](../../../../glossary.md#term-model) 0021C's pure five-input/five-output meaning and
[Compiler](../../../../glossary.md#term-compiler) 0005B's exact output-slot roles through one direct generated, [channel](../../../../glossary.md#term-channel)-owned, multi-pass
body that computes biased forward [variance](../../../../glossary.md#term-variance), correction-one running variance, explicit next
running statistics, saved [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) statistics, and normalized affine output without hidden state.

## [Mental model](../../../../glossary.md#term-mental-model)

```text
one explicit BATCH_NORM_TRAINING node
  -> validate five inputs, five distinct outputs, static channel geometry, and N >= 2
  -> assign each complete channel domain to exactly one scalar range
  -> exact mean pass -> corrected squared-deviation pass -> four channel-statistic stores
  -> normalized affine output pass
  -> publish five ordinary graph values directly; retain no state after the run
```

The range unit is one complete [channel](../../../../glossary.md#term-channel), not an input [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) or partial reduction. [Scalar](../../../../glossary.md#term-scalar) and
parallel-scalar execution therefore use the same arithmetic and [canonical](../../../../glossary.md#term-canonical-form) non-channel traversal.
Parallel ranges own disjoint channels and reuse one existing exact-sum scratch [slice](../../../../glossary.md#slice) per active
range. No partial [statistic](../../../../glossary.md#term-statistic), combine phase, cross-range [synchronization](../../../../glossary.md#term-synchronization), or state transition is
hidden in [Runtime](../../../../glossary.md#term-runtime).

## [Scope](../../../../glossary.md#term-scope)

- Recognize exactly one compiled `BatchNormKind.BATCH_NORM_TRAINING` [occurrence](../../../../glossary.md#term-occurrence) with
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
  into one CPU identity or runtime mode.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [inference glossary](../../../../glossary.md#term-inference); [CPU glossary](../../../../glossary.md#term-cpu); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Accept BFLOAT16, FLOAT32, and FLOAT64 independently at all five input positions. Promote in
  exact occurrence order: input with scale, then bias, running mean, and running variance. Require
  all five outputs, momentum, and epsilon to use the promoted result type.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [momentum glossary](../../../../glossary.md#term-momentum); [epsilon glossary](../../../../glossary.md#epsilon))</span>
- Revalidate [Compiler](../../../../glossary.md#term-compiler) 0005B's exact output [descriptors](../../../../glossary.md#term-descriptor) and [gradient](../../../../glossary.md#term-gradient)-eligibility roles:

  | Slot | [Shape](../../../../glossary.md#shape) | Required `requiresGrad` |
  |---|---|---|
  | 0 normalized output | exact input Shape | `input || scale || [bias](../../../../glossary.md#term-bias)` |
  | 1 next running [mean](../../../../glossary.md#term-mean) | exact `[C]` Shape | `input || runningMean` |
  | 2 next running [variance](../../../../glossary.md#term-variance) | exact `[C]` Shape | `input || runningVariance` |
  | 3 saved [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) mean | exact `[C]` Shape | `input` |
  | 4 saved inverse [standard deviation](../../../../glossary.md#term-standard-deviation) | exact `[C]` Shape | `input` |

- Require fully static resolved descriptors, input rank at least two, normalized non-negative
  channel axis within rank, four exact rank-one channel vectors, exact input/output Shape
  equality for slot zero, exact shared channel Shape for slots one through four, non-negative
  resolved layouts, and injective writable layouts for every output.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [vectors glossary](../../../../glossary.md#term-vector); [layouts glossary](../../../../glossary.md#layout); [injective glossary](../../../../glossary.md#term-injective))</span>
- Derive checked geometry before any declaration or [artifact](../../../../glossary.md#term-artifact) work:

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
  an unrepresentable CPU execution geometry.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- Use exactly one [channel](../../../../glossary.md#term-channel) range form. `start/end` is a half-open channel interval. Each owned
  channel traverses every non-channel coordinate in canonical prefix-major, then suffix-major,
  logical order for every pass. No range divides a channel domain.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Select [scalar](../../../../glossary.md#term-scalar) or bounded parallel-scalar execution during preparation:

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
  selected range count are cold preparation facts, not runtime tuning or generated kind dispatch.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Treat `channelCount == 0` as completely empty work even when non-channel extents are zero or
  one. Retain truthful input/output declarations, but perform no input/vector read, scratch use,
  generated invocation, output write, square root, or worker submission.<span data-glossary-navigation="true"> ([square root glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>
- Use one existing [alignment](../../../../glossary.md#term-alignment)-eight `AGGREGATE_EXACT_STATE` workspace slice per simultaneously
  active range when <span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>`channelCount > 0`. Size one slice from the promoted result computation format
  and exact `reductionCount` through the existing exact-sum state owner. Reuse that slice
  sequentially across channels in the range. Declare no second workspace, partial/combine state,
  materialization, saved-statistic scratch, or persistent resource.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [materialization glossary](../../../../glossary.md#materialization))</span>
- For each channel, execute the exact [finite](../../../../glossary.md#term-finite) algorithm and [operation](../../../../glossary.md#operation) order below:

  1. Read input values in canonical non-channel order, promote them to the result computation
     format, classify NaN/infinity, and obtain the exact represented sum through the existing
     fixed-width exact-sum state.
  2. Divide the exact sum by `N` and round once to the result computation format to produce the
     internal `batchMean`.
  3. Traverse the same domain again. Form each `deviation = input - batchMean` at the result
     computation boundary. Accumulate deviations and squared deviations in ordered compensated
     binary64 sums. Form the corrected numerator exactly in this order:<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>

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
     subtraction, multiplication, and addition at the result computation boundary:<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight))</span>

     ```text
     oneMinusMomentum   = 1 - momentum
     nextRunningMean    = oneMinusMomentum * runningMean
                          + momentum * batchMean
     nextRunningVariance = oneMinusMomentum * runningVariance
                           + momentum * unbiasedVariance
     ```

  6. Store slots one through four once for the channel in this order: next running mean, next
     running variance, saved batch mean, saved inverse standard deviation.
  7. Traverse the non-channel domain a third time and compute each slot-zero [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) exactly:

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
- Preserve [Model](../../../../glossary.md#term-model) 0021C's special-value classes:
  - any input NaN or infinity in a non-empty channel produces CPU canonical NaN for its mean,
    both variance estimates, saved inverse standard deviation, normalized output coordinates,
    and both next-statistic formulas after ordinary arithmetic;<span data-glossary-navigation="true"> ([canonical NaN glossary](../../../../glossary.md#term-canonical-nan))</span>
  - a finite constant domain, including signed zeros, produces positive-zero biased/unbiased
    variances; mean zero sign follows the exact selected sum/division rule;<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
  - old running variance is neither read for forward normalization nor validated, repaired, or
    clamped; it affects only slot two;<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
  - affine and old-[statistic](../../../../glossary.md#term-statistic) [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity) affect only their written formula roles;
  - momentum endpoints use the written formula, so zero coefficients do not suppress NaN or
    infinity;
  - finite overflow, underflow, signed zero, root, reciprocal, and affine behavior follow the
    frozen typed order above.<span data-glossary-navigation="true"> ([underflow glossary](../../../../glossary.md#term-underflow); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- Deduplicate repeated logical input [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values in first-occurrence order and retain an
  immutable five-position-to-boundary map. Input/input alias is legal read sharing. Require all
  five output <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>`ValueId` values to be distinct from every input and from one another.
- Declare each unique input once followed by all five outputs in semantic output order. The
  generated entry receives unique input carriers, five output carriers, one scratch segment,
  packed primitive geometry, and primitive `long start, long end`.
- Before mutation or worker submission, validate all ten semantic positions, deduplicated carrier
  compatibility, liveness and worker access, native order, alignment, complete spans, every
  output's writability/injectivity, the exact workspace, every workspace/buffer overlap, every
  output/input overlap, and every output/output overlap. Validation failure writes nothing and
  submits no work.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [liveness glossary](../../../../glossary.md#term-liveness); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [buffer glossary](../../../../glossary.md#term-buffer); [Validation glossary](../../../../glossary.md#term-validation))</span>
- After successful validation, store directly into all five ordinary output representations.
  There is no shadow result, commit phase, rollback, in-place running-statistic update, hidden
  cross-run state, or special saved-output lifetime. Arithmetic is non-throwing; an unrelated
  later generated/worker failure follows the existing Runtime contract and may expose already
  written output coordinates.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit); [rollback glossary](../../../../glossary.md#term-rollback); [run state glossary](../../../../glossary.md#run-state--runstate); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Support every unique boundary's exact `short[]`, `float[]`, `double[]`, or native-order
  [`MemorySegment`](../../../../glossary.md#term-memorysegment) carrier according to its own type, including all-heap, all-segment, and mixed
  patterns, arbitrary legal offsets, positive or zero input strides, and arbitrary injective
  output strides.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [strides glossary](../../../../glossary.md#term-stride))</span>
- Add one focused training IR, one arbitrary-axis training lowerer/geometry owner, one direct
  emitter, and one independent reference owner. Reuse current exact-sum, carrier, access, route,
  preparation, finalization, worker, overlap, artifact, and multi-output seams. Do not extend the
  inference IR into a training/inference union or create a generic normalization interpreter.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Emit one deterministic final, field-free, constructor-free generated class per specialization
  with exactly one public static typed <span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>`invoke` entry and no helper method.
- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly once from 49 to 50. The family, five-output
  signature, momentum/epsilon bits, range/pass/arithmetic identity, exact-state shape, boundary
  map, and generated body change compatibility. Schema 49 and earlier envelopes are safe misses;
  no migration reader or historical evidence mutation is added.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Schema glossary](../../../../glossary.md#term-schema))</span>
- Use a frozen optimal clean Java implementation with identical validation, channel ownership,
  exact-sum state, pass/dataflow shape, addressing, typed arithmetic, stores, and workspace reuse
  as the design, review, and performance oracle. Generated hot work may call only required JDK
  primitive/raw-bit, <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dataflow glossary](../../../../glossary.md#term-dataflow); [oracle glossary](../../../../glossary.md#term-test-oracle); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`Math.sqrt`, and typed `MemorySegment` operations and must call no Synaptik
  numerical, validation, or reference helper.

## Out of [scope](../../../../glossary.md#term-scope)

- `BATCH_NORM_INFERENCE` semantic or implementation changes; Layer/RMS/softmax/reduction changes;
  a shared normalization identity; decomposed recognition; inference/training mode flags; or
  folding inference and training artifacts together.<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [identity glossary](../../../../glossary.md#term-identity-meanings); [inference glossary](../../../../glossary.md#term-inference); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Stateful layers, parameters, buffers, assignment of next statistics, cross-step transport,
  training/evaluation mode, optimizer/session/checkpoint ownership, mutation, serialization, or
  hidden state.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [buffers glossary](../../../../glossary.md#term-buffer); [evaluation glossary](../../../../glossary.md#term-evaluation); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Compiler inference, deferred constraints, autograd formulas, gradient eligibility, saved-value
  capture/liveness, publication planning, or optimization changes. Slots three and four remain
  existing same-occurrence compiler auxiliaries, not backend-created tape values.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [gradient glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [liveness glossary](../../../../glossary.md#term-liveness); [publication glossary](../../../../glossary.md#term-publication); [planning glossary](../../../../glossary.md#term-planning); [occurrence glossary](../../../../glossary.md#term-occurrence); [backend glossary](../../../../glossary.md#backend))</span>
- Optional operands, default values, implicit constants, scalar/full-rank channel operands,
  optional momentum, batch counters, cumulative average, alternate momentum conventions,
  configurable correction, variance repair, or Tensor-valued scalars.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [channel glossary](../../../../glossary.md#term-channel); [momentum glossary](../../../../glossary.md#term-momentum); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [variance glossary](../../../../glossary.md#term-variance); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Dynamic or symbolic Shapes, unresolved layouts, runtime channel axes, rank below two, positive
  channels with reduction count below two, negative strides, non-injective outputs, or
  unrepresentable counts/spans/addresses/resources.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [runtime glossary](../../../../glossary.md#term-runtime); [axes glossary](../../../../glossary.md#term-axis); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>
- FLOAT16, integral, BOOL, complex, quantized, sparse, or unsigned arithmetic; configurable
  accumulation/result type; public casts; relaxed math; or cross-backend bitwise equality.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Partial channel reductions, reduction/combine trees, non-channel range splitting, atomic
  statistics, per-coordinate task submission, a denominator/statistic table, or more than one
  workspace.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- Vector API, native/vendor routes, fusion with adjacent work, materialization, autotuning,
  measured candidate selection, dynamic dispatch, or scalar-reference Runtime fallback.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Generated allocation, boxing, reflection, method handles, `invokedynamic`, dynamic constants,
  bootstrap methods, `Object` descriptors, collection/map/string lookup, or avoidable semantic,
  carrier, layout, output-role, pass, or kind dispatch in hot work.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [lookup glossary](../../../../glossary.md#term-lookup); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Public/shared API, Model, Compiler, Training, Runtime, Prepare, backend-contract, Trace,
  Planning, Engine, dependency, module-boundary, Gradle/toolchain, architecture, conformance, or
  integration changes beyond the CPU-private generator schema.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [conformance glossary](../../../../glossary.md#term-conformance); [schema glossary](../../../../glossary.md#term-schema))</span>
- Reopening historical [CPU](../../../../glossary.md#term-cpu) 0007A1D or creating a detailed CPU 0008 specification.

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
  validation, and generated execution.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [Shapes glossary](../../../../glossary.md#shape); [momentum glossary](../../../../glossary.md#term-momentum); [epsilon glossary](../../../../glossary.md#epsilon); [metadata glossary](../../../../glossary.md#term-metadata); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [gradients glossary](../../../../glossary.md#term-gradient); [lifetime glossary](../../../../glossary.md#term-lifetime); [admission glossary](../../../../glossary.md#term-admission); [lowering glossary](../../../../glossary.md#lowering))</span>
- [CPU](../../../../glossary.md#term-cpu) must analyze only the explicit `BATCH_NORM_TRAINING` node. Equivalent decomposed graphs
  remain ordinary nodes; <span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [graphs glossary](../../../../glossary.md#graph))</span>`BATCH_NORM_INFERENCE` retains its separate [schema](../../../../glossary.md#term-schema)-49 [identity](../../../../glossary.md#term-identity-meanings).
- The completed Model and Compiler contracts are implementation inputs, not files to revise. Any
  disagreement in signature, output order, descriptor roles, domain validity, promotion, scalar
  type, formula, or saved-value meaning is a stop condition.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [descriptor glossary](../../../../glossary.md#term-descriptor); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- CPU analysis chooses the route and declares every buffer/workspace requirement before shared
  assignment. CPU finalization may construct or reuse the schema-50 artifact only afterward and
  may not change the route or resource set. Runtime receives one immutable prepared executable.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [artifact glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- Planning continues to select CPU ownership only. Shared Prepare assigns existing buffer and
  workspace slots without interpreting channels, statistics, passes, momentum, or saved outputs.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Generated code must preserve the optimal clean Java specialized algorithm, hot-loop/dataflow
  shape, operation order, workspace reuse, and avoidable-overhead profile. A reference call,
  generic bridge, decomposed pointwise/reduction sequence, or runtime output-role/pass switch is
  not a specialized implementation.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [operation glossary](../../../../glossary.md#operation); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Any required public/shared/build/architecture/conformance/integration change, new resource
  kind, second workspace, partial/combine representation, unresolved semantic choice, or path/type
  ceiling violation is a stop/replan condition.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>

## Current-code evidence

- [Model](../../../../glossary.md#term-model) 0021C and current `TensorBatchNormTrainingExpressions` fix the five ordered inputs, five
  ordered outputs, exact Shapes, ordered promotion, typed momentum/epsilon, <span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [momentum glossary](../../../../glossary.md#term-momentum); [epsilon glossary](../../../../glossary.md#epsilon))</span>`C == 0 || N >= 2`,
  metadata, saved-statistic meanings, formulas, and pure transition boundary.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- Current `ReductionNormalizationInference` revalidates all five descriptors and deferred domain
  obligation. <span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>`NormalizationGradientRules` retrieves exact same-occurrence slots three/four and
  assigns public-output gradients by exact output slot. Neither decomposes the forward node.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [gradients glossary](../../../../glossary.md#term-gradient); [node glossary](../../../../glossary.md#node))</span>
- `CpuCapabilityProvider` and `CpuPartitionLowering` admit only batch inference today; training
  therefore fails closed.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [inference glossary](../../../../glossary.md#term-inference))</span>
- [Schema](../../../../glossary.md#term-schema)-49 `CpuBatchNormInferenceIr`, lowering, emitter, reference, preparation, and execution
  prove arbitrary-axis geometry, ordered mixed floating carriers, first-occurrence input
  deduplication, static layout checks, direct generated invocation, and cold range selection. Its
  zero-workspace coordinatewise algorithm is not the training algorithm and remains separate.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [axis glossary](../../../../glossary.md#term-axis); [layout glossary](../../../../glossary.md#layout))</span>
- `CpuStatisticalReductionEmitter`, `CpuExactSumEmitter`, advanced-reduction geometry, and Layer
  normalization already prove exact-sum per-range scratch, corrected squared-deviation arithmetic,
  workspace slicing, and direct multi-pass generated bodies. Training reuses these narrow seams
  but owns one separate five-output emitter and geometry.<span data-glossary-navigation="true"> ([Layer normalization glossary](../../../../glossary.md#layer-normalization))</span>
- Ordering and dropout already prove direct multi-output boundary declaration and invocation.
  Existing generic overlap logic handles at most their current shapes; training must explicitly
  validate all five output/input and ten output/output pairs rather than inheriting a one/two-
  output assumption.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- Existing preparation has one optional [CPU](../../../../glossary.md#term-cpu) workspace and `AGGREGATE_EXACT_STATE`; no shared
  resource kind or contract change is needed for one exact-state slice per selected range.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>

## Package impact

Existing [CPU](../../../../glossary.md#term-cpu)-private packages changed:

- `io.github.pho001.synaptik.backend.cpu` — exact truthful capability [admission](../../../../glossary.md#term-admission).
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — [immutable](../../../../glossary.md#term-immutable) training [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — arbitrary-axis channel-domain
  geometry, boundary mapping, and exact-state sizing.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [channel glossary](../../../../glossary.md#term-channel))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct three-pass/five-output
  generated body.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent differential [oracle](../../../../glossary.md#term-test-oracle).
- Existing `route.portable`, [`prepare`](../../../../glossary.md#prepare), `executable`, and [`cache`](../../../../glossary.md#term-cache) packages — scalar realization,
  declaration, workspace assignment, complete cold validation, invocation, and schema-50
  compatibility.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [schema glossary](../../../../glossary.md#term-schema))</span>

Packages added: None.

Type placement:

- Add `CpuBatchNormTrainingIr` in `internal.ir` for exact five-position input types, result type,
  raw momentum/epsilon bits, rank/channel axis, algorithm/pass/arithmetic version, exact-state
  shape, position map, and ten ordered access plans.<span data-glossary-navigation="true"> ([momentum glossary](../../../../glossary.md#term-momentum); [epsilon glossary](../../../../glossary.md#epsilon); [rank glossary](../../../../glossary.md#term-rank); [shape glossary](../../../../glossary.md#shape))</span>
- Add `CpuBatchNormTrainingLowering` in `internal.lowering` for static
  prefix/channel/suffix/reduction geometry, five output layouts, unique input boundaries, exact
  state size, canonical traversal, and packed cold geometry.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Add `CpuBatchNormTrainingEmitter` in `internal.codegen.emit` for the sole direct exact-mean,
  corrected-variance, transition, saved-statistic, and normalized-output body.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- Add `CpuBatchNormTrainingReferenceKernel` in `internal.reference`; it independently derives
  channel domains and expected five-output values without production lowering, emission,
  binding, or generated helpers.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding))</span>
- Add no second emitter, validator type, state owner, generic normalization facade, manager,
  registry, or training/inference union.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [normalization glossary](../../../../glossary.md#term-normalization); [facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry); [inference glossary](../../../../glossary.md#term-inference))</span>

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- `CpuCapabilityProvider.java` and [CPU](../../../../glossary.md#term-cpu) package Javadoc;
- `internal/ir/CpuPortableKernelIr.java`, new `CpuBatchNormTrainingIr.java`, and [IR](../../../../glossary.md#term-intermediate-representation-ir) package Javadoc;
- `internal/lowering/CpuPartitionLowering.java`, new `CpuBatchNormTrainingLowering.java`, and
  lowering package Javadoc;<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new
  `CpuBatchNormTrainingEmitter.java`, existing exact-sum/carrier seams only where required, and
  emitter package Javadoc;<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `internal/cache/CpuGeneratorSchema.java`, `CpuKernelSpecialization.java` only if exact scratch or
  entry validation requires it, and cache package Javadoc;<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [cache glossary](../../../../glossary.md#term-cache))</span>
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
  change; and<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization); [prepare glossary](../../../../glossary.md#prepare))</span>
- the historical pointwise-ledger test only for its current-schema assertion; retained historical
  evidence resources remain immutable.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [schema glossary](../../../../glossary.md#term-schema); [immutable glossary](../../../../glossary.md#term-immutable))</span>

Expected documentation/[planning](../../../../glossary.md#term-planning) paths in the implementation change:

- `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  global roadmap. Tensor/Compile/Training APIs, Model/Compiler plans, architecture, shared Prepare,
  backend-contract, conformance, integration, and build files are review-only unless a stop
  condition is reached.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

The complete implementation, tests, Javadocs, and documentation may modify or create at most 48
repository paths: at most 28 production/[Javadoc](../../../../glossary.md#term-javadoc) paths, 15 test paths, and exactly the five named
documentation/[planning](../../../../glossary.md#term-planning) paths. Exactly four new [CPU](../../../../glossary.md#term-cpu)-private production types are permitted: the
[IR](../../../../glossary.md#term-intermediate-representation-ir), lowerer, emitter, and independent reference [owner](../../../../glossary.md#term-owner) listed above.

This exceeds the planning guide's ordinary guardrail because one truthful five-output generated
family must update the current [sealed](../../../../glossary.md#term-sealed) IR, [lowering](../../../../glossary.md#lowering), generator, [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache), route, one-workspace
preparation, finalization, executable [binding](../../../../glossary.md#term-binding), package-Javadoc, inventory, and differential seams
atomically. The behavioral core remains four focused types in one [module](../../../../glossary.md#term-module). Splitting declaration,
multi-output binding, or execution would temporarily advertise a family that cannot publish all
of its semantic outputs and would duplicate the same module/evidence gate.

A 49th path, fifth new production type, second emitter/validator, new resource kind, second
workspace, public/shared/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration path, [inference](../../../../glossary.md#term-inference) semantic
change, partial/combine route, or CPU 0008 behavior is a stop/replan condition. An omitted expected
path does not authorize an unrelated path.

## Detailed generated-code design boundary

- `CpuBatchNormTrainingIr` is instruction-free. Its cache identity includes family, five semantic
  input types, result type, raw momentum/epsilon bits, rank/channel axis, algorithm/arithmetic/pass
  version, exact-state limb/slice shape, five-position map, ordered unique-input/five-output access
  plans, and scratch-bearing entry shape.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [momentum glossary](../../../../glossary.md#term-momentum); [epsilon glossary](../../../../glossary.md#epsilon); [rank glossary](../../../../glossary.md#term-rank); [channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [slice glossary](../../../../glossary.md#slice); [shape glossary](../../../../glossary.md#shape))</span>
- `CpuBatchNormTrainingLowering.Geometry` retains concrete static extents/layouts,
  prefix/channel/suffix/reduction/input counts, output positions, canonical traversal, and exact
  scratch size. Concrete compatible extents, strides, offsets, and selected range count stay cold
  unless they change exact-state width or emitted bytes.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [canonical glossary](../../../../glossary.md#term-canonical-form); [strides glossary](../../../../glossary.md#term-stride))</span>
- The generated outer loop owns channels. Each channel performs exact-sum pass, corrected-variance
  pass, four scalar statistic stores, and normalized-output pass. Dense arrays may use cold-proved
  integer indexes; arbitrary layouts use primitive odometers/increments. No hot division/modulo
  or full coordinate reconstruction is repeated per element when a direct increment is provable.<span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance); [scalar glossary](../../../../glossary.md#term-scalar); [statistic glossary](../../../../glossary.md#term-statistic); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [element glossary](../../../../glossary.md#term-element))</span>
- One specialization has one generated artifact. The existing four-candidate/one-artifact/
  zero-unroll-variant budget remains unchanged. Training is never materialized or Vector-selected.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [Vector glossary](../../../../glossary.md#term-vector))</span>
- Schema 50 is current-only. Schema-49 and earlier envelopes safely miss and regenerate; no
  compatibility alias, migration reader, or retained-evidence rewrite is added.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>

## Acceptance criteria

- Capability admits exactly first-class `BATCH_NORM_TRAINING` with `BatchNormTrainingAttrs`, five
  inputs, five outputs, all current BFLOAT16/FLOAT32/FLOAT64 promotion combinations, exact typed
  momentum/epsilon, static Shapes/layouts, `C == 0 || N >= 2`, and exact output [descriptors](../../../../glossary.md#term-descriptor).
- `BATCH_NORM_INFERENCE` remains separately admitted and unchanged. Source-backed tests prove an
  equivalent decomposed graph remains ordinary operations and never becomes either batch IR.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Lowering records exact types/scalars/rank/axis/counts, five-position input map, unique boundaries,
  five distinct outputs, access plans, exact-state shape, and channel range identity. Checked
  geometry and resource validation complete before declarations.<span data-glossary-navigation="true"> ([Lowering glossary](../../../../glossary.md#lowering); [scalars glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [channel glossary](../../../../glossary.md#term-channel); [identity glossary](../../../../glossary.md#term-identity-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Semantic/differential coverage includes channel axis first/middle/last; rank two and higher;
  channel extent zero/one/larger; reduction count two and larger; rejected positive-channel
  counts zero/one; same/mixed types; dense/general/zero-stride inputs; arbitrary injective output
  layouts; offsets; repeated inputs; all heap/all segment/mixed carriers; and mixed output Shapes.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [stride glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective); [heap glossary](../../../../glossary.md#term-heap))</span>
- Formula tests lock exact sum/divide mean, corrected squared-deviation numerator, separately
  divided biased/unbiased variances, epsilon placement, reciprocal square root, new-batch-weight
  transitions, statistic store roles/order, saved values, and multiply-based normalized affine
  order with the exact FLOAT32/FLOAT64 boundaries.<span data-glossary-navigation="true"> ([square root glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); [weight glossary](../../../../glossary.md#term-weight); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- The [Model](../../../../glossary.md#term-model) `[1,2,3]` example with scale `2`, [bias](../../../../glossary.md#term-bias) `0.5`, old [mean](../../../../glossary.md#term-mean) `10`, old [variance](../../../../glossary.md#term-variance) `4`, [momentum](../../../../glossary.md#term-momentum)
  `0.25`, and [epsilon](../../../../glossary.md#epsilon) `1e-5` produces mean `2`, biased variance `2/3`, unbiased variance `1`, next
  mean `8`, next variance `3.25`, saved inverse [standard deviation](../../../../glossary.md#term-standard-deviation)
  `1/sqrt(2/3+1e-5)`, and normalized output approximately
  `[-1.949471, 0.5, 2.949471]` under the selected typed [tolerance](../../../../glossary.md#term-tolerance).
- Special-value tests cover input/affine/old-statistic NaNs and infinities, both signed zeros,
  finite constants, negative old variance, momentum <span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [finite glossary](../../../../glossary.md#term-finite); [constants glossary](../../../../glossary.md#term-constant))</span>`-0`, `+0`, and `1`, subnormals,
  overflow/underflow, positive-zero variance, saved inverse standard deviation, and unsuppressed
  zero-coefficient NaN/infinity.<span data-glossary-navigation="true"> ([subnormals glossary](../../../../glossary.md#term-subnormal); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- An independent StrictMath/high-precision oracle checks ordinary finite results within 16 result
  ulps for FLOAT64, four result ulps for FLOAT32, and one represented BFLOAT16 ulp. These are the
  existing conservative maxima already justified by CPU 0007D's corrected variance and CPU
  0007F's normalization evidence, the two numerical components composed here; this task does not
  invent a looser family-specific threshold. Exact special classes, infinities, and zero signs use
  class/raw-sign assertions. Generated and frozen identical-algorithm direct Java are raw-bit
  equal for all five outputs.<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [oracle glossary](../../../../glossary.md#term-test-oracle); [CPU glossary](../../../../glossary.md#term-cpu); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Resource declarations contain unique inputs in first-use order, then five outputs in semantic
  order, plus exactly one alignment-eight exact-state workspace when work is non-empty. They
  contain no materialization, second workspace, partial/combine value, hidden saved buffer, or
  persistent state.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [materialization glossary](../../../../glossary.md#materialization); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Empty channel work performs no reads, scratch use, square root, generated call, store, or worker
  submission and preserves every carrier and canary.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [canary glossary](../../../../glossary.md#term-canary))</span>
- Cold binding validates the complete carrier/span/layout/workspace/overlap matrix before mutation
  or submission. All input/input aliases remain legal read sharing; all output/input and
  output/output overlaps fail without writes or submission.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Scalar and parallel-scalar results are raw-bit deterministic across complete-channel ranges,
  worker counts, repeated runs, and concurrent runs. Each non-empty channel is owned once; each
  statistic coordinate is written once; each normalized coordinate is written once; every active
  range receives a disjoint exact-state slice.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [slice glossary](../../../../glossary.md#slice))</span>
- Generated classes are deterministic, final, field-free, constructor-free, and expose exactly
  one typed static `invoke` with unique input carriers, five output carriers, scratch, geometry,
  and `long start/end`. There is no `Object`, bridge descriptor, kind/pass/output-role argument,
  validation/reference call, or helper method.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- Complete `javap -c -p` and `javap -v -p` inspection proves exact pass and store order, direct
  typed carrier/address work, generation-time family/type/axis/carrier specialization, and absence
  of allocation, boxing, reflection, method handles, <span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>`invokedynamic`, dynamic constants,
  bootstrap methods, collections/maps/strings, semantic dispatch, and avoidable address work.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- A complete member allowlist permits only required primitive/raw-bit operations, `Math.sqrt`, and
  typed [`MemorySegment`](../../../../glossary.md#term-memorysegment) access. Any unexpected member is a failure; generated classes reference no
  Synaptik numerical, validation, exact-sum runtime, or reference helper.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- Specialization/cache identity distinguishes every byte-shaping fact named above. Compatible
  cold bases, offsets, ranges, slots, workers, graph/run identities, and values do not change
  identity. Different raw momentum or epsilon bits are incompatible.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [run glossary](../../../../glossary.md#term-run))</span>
- Schema advances exactly 49 to 50. Older envelopes safely miss; completed task records and
  retained evidence stay immutable and historical.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Freeze exactly eight training [targets](../../../../glossary.md#term-target). `S` means a matching native-order `MemorySegment`.
  Vector layout pairs are `(offset,stride)`. `BNT-F64-A1` uses usable parallelism one to measure
  the scalar range form; the other seven use usable parallelism four. Every target uses
  `minimumElementsPerWorker = 4096`; parallel targets produce four channel ranges unless the exact
  channel count makes fewer ranges truthful.

  | Target | [Shape](../../../../glossary.md#shape) / [axis](../../../../glossary.md#term-axis) | Inputs -> result | Carriers | [Layout](../../../../glossary.md#layout) |
  |---|---|---|---|---|
  | `BNT-BF16-A1` | `[32,64,256]` / 1 | all [BFLOAT16](../../../../glossary.md#term-bfloat16) -> BFLOAT16 | all `short[]` | dense |
  | `BNT-F32-A1` | `[32,64,256]` / 1 | all [FLOAT32](../../../../glossary.md#term-float32) -> FLOAT32 | all `float[]` | dense |
  | `BNT-F64-A1` | `[32,64,256]` / 1 | all [FLOAT64](../../../../glossary.md#term-float64) -> FLOAT64 | all `double[]` | dense |
  | `BNT-F32-A0` | `[128,4096]` / 0 | all FLOAT32 -> FLOAT32 | all `float[]` | dense |
  | `BNT-F32-A2` | `[32,256,64]` / 2 | all FLOAT32 -> FLOAT32 | all `float[]` | dense |
  | `BNT-MIX-F64` | `[16,32,64]` / 1 | FLOAT32/FLOAT64/BFLOAT16/FLOAT32/FLOAT64 -> FLOAT64 | mixed arrays/`S` -> five mixed FLOAT64 outputs | frozen legal general layouts |
  | `BNT-MIX-F32` | `[16,32,64]` / 1 | BFLOAT16/FLOAT32/BFLOAT16/FLOAT32/BFLOAT16 -> FLOAT32 | mixed arrays/`S` -> five mixed FLOAT32 outputs | frozen legal general layouts |
  | `BNT-REPEAT` | `[32,64,128]` / 1 | all FLOAT32 -> FLOAT32 | repeated exact [vector](../../../../glossary.md#term-vector) input boundary; mixed outputs | dense input, legal strided vectors/statistics |

  Freeze unchanged FLOAT32 [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [inference](../../../../glossary.md#term-inference), Layer [Norm](../../../../glossary.md#term-norm), `VARIANCE`, and [pointwise](../../../../glossary.md#term-pointwise) `ADD` controls
  at their retained CPU-0007F1/0007F/0007D/0005 definitions. This matrix is proportional to the
  new family: all result formats, three channel-axis positions, dense/general/mixed carriers,
  repeated inputs, five-output stores, exact-state scratch, and both scalar/parallel code paths
  are represented without claiming universal layout or workload performance.
- Every target and control passes generated/direct `<= 1.15x` in each of five accepted isolated
  forks and for the median of fork medians. The frozen optimal clean Java oracle performs
  identical useful validation, arithmetic, passes, addressing, conversions, workspace reuse, and
  stores and is never slowed, padded, or routed through production helpers. Reject and retain a
  whole sample when any semantic, checksum, environment, target, or control gate fails.<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median); [fork glossary](../../../../glossary.md#term-fork-meanings); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [checksum glossary](../../../../glossary.md#term-checksum))</span>
- Focused tests, one final uncached CPU suite, Javadoc, retained semantic/Class-File/decompilation/
  member/operation-count/checksum/five-fork evidence, exact scope, schema/status/staging, and
  whitespace gates pass. A distinct clean documentation context finalizes affected Javadocs, CPU
  guide, glossary impact, task, master plan, and roadmap without repeating stable Java/performance
  work.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [decompilation glossary](../../../../glossary.md#term-decompilation); [operation glossary](../../../../glossary.md#operation); [scope glossary](../../../../glossary.md#term-scope))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

The implementation context runs focused tests after stabilization, then one final [CPU](../../../../glossary.md#term-cpu) suite:

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

Retain one [immutable](../../../../glossary.md#term-immutable) evidence bundle outside the repository containing exact source/probe inputs,
environment/[JVM](../../../../glossary.md#term-java-virtual-machine-jvm) facts, all eight [targets](../../../../glossary.md#term-target) and four controls, specializations, complete
`javap -c -p` and `javap -v -p`, member and [operation](../../../../glossary.md#operation)-count reports, semantic/accuracy/range/
workspace/[alias](../../../../glossary.md#aliasing-and-arena)/[canary](../../../../glossary.md#term-canary) results, five isolated `-Xms1g -Xmx1g` [fork](../../../../glossary.md#term-fork-meanings) outputs, rejected whole
[samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), summaries, inventory, and a [SHA-256](../../../../glossary.md#term-sha-256) [manifest](../../../../glossary.md#term-manifest). Each fork uses at least five randomized
[warmup](../../../../glossary.md#term-warmup) [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), nine randomized measured rounds, adaptive batches of at least 25 ms, deterministic
inputs, randomized generated/direct order, and raw/[checksum](../../../../glossary.md#term-checksum) verification.

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
exact path/type ceilings, [schema](../../../../glossary.md#term-schema) 50 and historical-evidence wording, CPU 0007A1D Review needed,
CPU 0007F/0007F1 Complete, CPU 0007F2 Complete only after every gate, CPU 0008 and later tasks
Draft without detailed specifications, and empty staging.

Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration suites remain deferred to CPU
0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci) because the implementation changes one [concrete backend](../../../../glossary.md#concrete-backend) without a
shared boundary. A stop condition that changes a shared boundary requires replanning and
proportionate validation.

## Dependencies

- Complete Model 0021C owns the exact first-class five-input/five-output meaning, arbitrary channel
  axis, Shapes, promotion, typed scalars, domain obligation, biased/unbiased statistics, saved
  values, transition, metadata, and special-value contract.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [axis glossary](../../../../glossary.md#term-axis); [Shapes glossary](../../../../glossary.md#shape); [scalars glossary](../../../../glossary.md#term-scalar); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Complete Compiler 0005B and current inference/preflight/gradient code own exact five-output
  capture and validation, output-slot roles, same-occurrence saved slots three/four, and
  first-class identity. CPU changes no formula or gradient behavior.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [preflight glossary](../../../../glossary.md#term-preflight); [gradient glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Complete CPU 0007F supplies exact-sum per-range workspace, mixed-floating normalization
  carriers, multi-pass generated evidence, and workspace slicing. Complete CPU 0007F1 supplies
  arbitrary-axis batch geometry, five-position input deduplication, static admission, direct
  generated batch execution, and schema-49 compatibility. Neither existing IR is reused as the
  training semantic identity.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [admission glossary](../../../../glossary.md#term-admission); [schema glossary](../../../../glossary.md#term-schema); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir))</span>
- Existing ordering/dropout multi-output paths and shared Prepare/Runtime contracts carry the
  five ordinary output boundaries and one workspace unchanged. CPU 0007A1D remains historical
  Review needed and is not a dependency.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [carry glossary](../../../../glossary.md#term-carry); [dependency glossary](../../../../glossary.md#term-dependency))</span>

All dependencies and semantic/resource decisions are settled. No architecture or shared-contract
blocker is known within the [fail-closed](../../../../glossary.md#term-fail-closed) static complete-[channel](../../../../glossary.md#term-channel) [scope](../../../../glossary.md#term-scope).

## Follow-up tasks

- CPU 0008 remains Draft for the remaining heavy portable families and keeps its corrected
  Model/Compiler prerequisites. No detailed 0008 specification is created here.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- CPU 0008C must preserve the prohibition on recognizing decomposed normalization as a first-class
  semantic kernel.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [kernel glossary](../../../../glossary.md#kernel))</span>
- CPU 0009 or continuous integration owns repository-wide portable coverage, conformance, and
  integration closure.<span data-glossary-navigation="true"> ([continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [CPU](../../../../glossary.md#term-cpu) 0007A1D remains historical Review needed; its failed local evidence gate is independent.

## Architecture impact

Expected impact: None.

This task adds one [CPU](../../../../glossary.md#term-cpu)-private generated forward family behind unchanged [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare),
and [Runtime](../../../../glossary.md#term-runtime) contracts. If implementation requires a public/shared numerical contract, [dependency](../../../../glossary.md#term-dependency),
[module](../../../../glossary.md#term-module), resource kind, architecture rule, or build change, stop and report rather than editing
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
must inspect the actual diff and retained evidence. It must apply General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend)
Guide, [Planning](../../../../glossary.md#term-planning), and Example profiles as relevant; finalize all affected Javadocs, the [CPU](../../../../glossary.md#term-cpu) guide,
glossary impact, this task, master plan, and roadmap; inspect rendered Javadocs; reuse successful
Java/performance evidence unless executable behavior changes; and record reasoned no-change
conclusions for [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) contracts, architecture/ADRs/tests,
shared [Prepare](../../../../glossary.md#prepare)/backend-contract/[Runtime](../../../../glossary.md#term-runtime), [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), other [modules](../../../../glossary.md#term-module), and the
accepted CPU 0008 ordering correction.

## Local decisions

- Use a separate training IR/lowerer/emitter/reference quartet. Training's reductions, exact-state
  workspace, five outputs, and statistic transitions are not inference identity or trailing
  Layer/RMS geometry.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [statistic glossary](../../../../glossary.md#term-statistic); [inference glossary](../../../../glossary.md#term-inference); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Own complete channels in scalar and parallel-scalar ranges. This keeps one deterministic
  reduction order and eliminates partial/combine state. Small channel counts may underuse workers;
  that limitation is safer than introducing unplanned synchronization or workspace topology.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [channel glossary](../../../../glossary.md#term-channel); [synchronization glossary](../../../../glossary.md#term-synchronization); [topology glossary](../../../../glossary.md#term-topology))</span>
- Reuse one exact-sum scratch slice per active channel range. The saved outputs are ordinary graph
  buffers, not workspace, persistent prepared state, or a runtime tape.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [graph glossary](../../../../glossary.md#graph); [buffers glossary](../../../../glossary.md#term-buffer); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Use exact sum for the mean and the completed corrected two-pass variance precedent. Compute
  biased and unbiased results by separate division from one corrected numerator. This avoids the
  unstable raw-moment formula and does not derive one estimate from the other.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean); [variance glossary](../../../../glossary.md#term-variance))</span>
- Preserve multiplication by saved inverse standard deviation and explicit multiply-then-add
  affine order. Preserve written momentum arithmetic even at zero/one endpoints.<span data-glossary-navigation="true"> ([standard deviation glossary](../../../../glossary.md#term-standard-deviation); [momentum glossary](../../../../glossary.md#term-momentum))</span>
- Keep compatible concrete extents/layouts cold except where exact-state width changes generated
  bytes. Bake family, types, raw scalars, axis/rank, algorithm/pass/state shape, boundary map,
  access/carriers, and entry shape.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [axis glossary](../../../../glossary.md#term-axis); [rank glossary](../../../../glossary.md#term-rank); [shape glossary](../../../../glossary.md#shape))</span>
- Advance compatibility exactly from schema 49 to 50. Older envelopes are safe misses and require
  no migration reader.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>

## Known limitations

- Only fully static resolved-layout scalar/parallel-scalar forward execution is planned. Dynamic
  Shapes, split reductions, Vector/native routes, fusion, materialization, and hidden state remain
  unsupported.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [Shapes glossary](../../../../glossary.md#shape); [Vector glossary](../../../../glossary.md#term-vector); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization))</span>
- Parallelism is bounded by channel count because each reduction domain has one owner. The task
  makes no claim that channel-poor workloads saturate the CPU.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Finite results follow the selected CPU algorithm and tolerance matrix. Raw-bit identity is
  required only between generated and identical-algorithm direct Java, not across backends.<span data-glossary-navigation="true"> ([Finite glossary](../../../../glossary.md#term-finite); [tolerance glossary](../../../../glossary.md#term-tolerance); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [backends glossary](../../../../glossary.md#backend))</span>
- The performance gate covers only the frozen proportional matrix and recorded environment; it is
  not a general throughput, tuning, or layout-optimality claim.<span data-glossary-navigation="true"> ([throughput glossary](../../../../glossary.md#term-throughput))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `01a03a7a-07e6-7473-a5fd-5f263df4236d` and resumed performance context
  `01a03c47-6f66-7372-84bf-b8937d492186` supplied the stabilized executable evidence. The final
  focused matrix passed 10 suites/114 tests with zero failures, errors, or skips. The final
  uncached <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>`./gradlew :backends:cpu:test --rerun-tasks` passed 86 suites/472 tests with zero
  failures or errors and two expected skips. Documentation-only Java changes followed, so clean
  documentation context `01a03c62-6576-7df0-bb6b-5f454918fb72` did not repeat either Java suite.
- The retained bundle contains eight deterministic [schema](../../../../glossary.md#term-schema)-50 generated classes plus complete
  `javap -c -p` and `javap -v -p`, member allowlists, forbidden-reference scans, and operation
  counts. All eight classes use direct odometer traversals, one <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`Math.sqrt` site, and four semantic
  division sites, with no allocation, boxing, reflection, method handle, <span data-glossary-navigation="true"> ([method handle glossary](../../../../glossary.md#term-method-handle))</span>`invokedynamic`, dynamic
  constant, collection/map/string dispatch, invocation-time layout construction, or Synaptik
  numerical/reference helper.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [dispatch glossary](../../../../glossary.md#term-dispatch); [layout glossary](../../../../glossary.md#layout))</span>
- Five accepted isolated Java 26.0.1 `-Xms1g -Xmx1g` forks passed all 60 per-fork and 12
  aggregate generated/optimal-clean-Java ratios at `<= 1.15x`. The worst [fork](../../../../glossary.md#term-fork-meanings) ratio was
  `1.123421082x` and the worst aggregate ratio was `1.110678870x`, both `BNT-REPEAT`. One whole
  environment/classpath sample was rejected before measurement and retained; no measured ratio
  sample was discarded.<span data-glossary-navigation="true"> ([classpath glossary](../../../../glossary.md#term-classpath); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- The read-only retained evidence root is
  `/private/tmp/synaptik-cpu-0007f2-retained-evidence-20260825`. Both `shasum -a 256 -c
  SHA256SUMS` and `shasum -a 256 -c SHA256SUMS.sha256` passed. The [manifest](../../../../glossary.md#term-manifest) [digest](../../../../glossary.md#term-digest) is
  `d1138b75924cea2b1bbce6ba127213eb2181de7156884b857ce1e475b9b95edb`.
- Clean documentation context `01a03c62-6576-7df0-bb6b-5f454918fb72` applied the General,
  API/Javadoc, Backend Guide, and Planning profiles. It finalized every new training-type
  contract, the changed schema/plan/executable contracts, the CPU guide, glossary, this task, CPU
  master plan, and roadmap. <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Backend glossary](../../../../glossary.md#backend); [Planning glossary](../../../../glossary.md#term-planning))</span>`./gradlew :backends:cpu:javadoc` passed after final Javadoc edits;
  generated pages for the four new types and changed constructors/records were inspected. Local
  Markdown links and anchors, fences, final newlines, whitespace, status, schema, path/type/test,
  staging, and retained-manifest checks passed. `git diff --check` and
  `git diff --cached --check` passed; staging remained empty.
- One final sandboxed Javadoc invocation was denied access to Gradle's user-home wrapper lock
  before Gradle started. The immediate approved <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`./gradlew :backends:cpu:javadoc` rerun passed
  with only the two expected incubating Vector API warnings; this was an environment-access retry,
  not a source or test failure.<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>
- Final dirty-worktree count is 47 paths: 25 production/Javadoc, 15 test, five named 0007F2
  documentation/planning paths, and the two preserved unrelated NN/Engine planning paths. The
  bounded 0007F2 scope is therefore 45 paths, within 48 overall, 28 production/Javadoc, and 15
  test ceilings, with exactly four new CPU-private production types and exactly the five required
  task documentation paths.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine); [scope glossary](../../../../glossary.md#term-scope))</span>
- Status checks confirm [CPU](../../../../glossary.md#term-cpu) 0007A1D remains `Review needed`; CPU 0007F, 0007F1, and 0007F2 are
  `Complete`; CPU 0008 and later rows remain `Draft` without detailed specifications; and the
  Conv1d/Conv3d-before-general-DAG ordering correction remains intact.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>

## Implementation notes

- Added one separate training IR, lowerer/geometry owner, direct emitter, and independent
  clean-Java reference owner. Existing inference identity and semantics remain unchanged.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [owner glossary](../../../../glossary.md#term-owner); [inference glossary](../../../../glossary.md#term-inference); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Capability, lowering, preparation, finalization, and executable binding now carry exactly one
  explicit five-input/five-output static occurrence. Repeated inputs share first-occurrence read
  boundaries; five outputs remain distinct and are validated against every input, output, and the
  exact workspace before mutation or worker submission.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding); [carry glossary](../../../../glossary.md#term-carry); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Complete-channel scalar and parallel-scalar ranges reuse one exact-sum state slice per active
  range. The generated body preserves the exact mean, corrected squared-deviation numerator,
  separate biased/unbiased divisions, biased saved inverse standard deviation, new-batch-weight
  momentum transitions, four statistic stores, and multiply-based normalized affine output.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [scalar glossary](../../../../glossary.md#term-scalar); [slice glossary](../../../../glossary.md#slice); [mean glossary](../../../../glossary.md#term-mean); [standard deviation glossary](../../../../glossary.md#term-standard-deviation); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [weight glossary](../../../../glossary.md#term-weight); [momentum glossary](../../../../glossary.md#term-momentum); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- Schema advanced exactly once from 49 to 50. Earlier envelopes remain safe misses and retained
  historical evidence was not modified.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema))</span>
- No architecture or ADR update is needed: the implementation stays inside the concrete CPU
  backend's existing analysis/finalization/runtime boundary. No public Tensor, Compile, Training,
  Engine, Runtime, Prepare, backend-contract, Trace, Planning, conformance, integration, Gradle,
  dependency, or module-boundary contract changed. Existing Model 0021C and Compiler 0005B
  contracts remain accurate. Nine affected CPU package Javadocs now include the training family
  and current schema while preserving their package purpose, visibility boundary, and unsupported
  internal-surface warning. CPU package inventory tests cover the four new private owners.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [Planning glossary](../../../../glossary.md#term-planning); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler))</span>

## Completion summary

- Completed changes: implemented and documented portable schema-50 first-class batch-normalization
  training with explicit five-input/five-output semantics, complete-channel deterministic ranges,
  exact-state workspace, saved statistics, and pure running-statistic transitions.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization); [channel glossary](../../../../glossary.md#term-channel); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- Files changed or created: 25 production/Javadoc paths, 15 test paths, and five 0007F2
  documentation/planning paths; two unrelated pre-existing NN/Engine planning edits were preserved
  unchanged.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Tests and validation: focused 10 suites/114 tests and final CPU 86 suites/472 tests passed;
  eight-class semantic/Class-File/member/operation-count evidence and all five-fork performance
  gates passed; final CPU Javadoc, Markdown, manifest, scope, status, staging, and whitespace gates
  passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [operation glossary](../../../../glossary.md#operation); [fork glossary](../../../../glossary.md#term-fork-meanings); [manifest glossary](../../../../glossary.md#term-manifest); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean context `01a03c62-6576-7df0-bb6b-5f454918fb72` finalized
  affected Javadocs, CPU guide, glossary, task, master plan, and roadmap without changing
  executable behavior or repeating stable Java/performance work.
- Documentation impact: the CPU guide now explains the executable training lifecycle and bounded
  evidence; planning records identify 0007F2 as Complete and CPU 0008 as the next Draft frontier.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Javadoc review: all four new types, nine affected packages, and changed
  record/constructor/schema contracts describe semantics, parameters, results, ownership,
  mutation, resources, and failures where applicable.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: batch-normalization training and generated-artifact entries now distinguish
  biased saved statistics, unbiased running transition, complete-channel exact-state execution,
  and schema 50 from inference and hidden state.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [inference glossary](../../../../glossary.md#term-inference))</span>
- Unresolved issues: None.
- Follow-up required: None. [CPU](../../../../glossary.md#term-cpu) 0008 remains separate Draft work.

Status: Complete

## Status gate

Do not change the task, [CPU](../../../../glossary.md#term-cpu) master plan, or roadmap to `Complete` until every acceptance criterion,
the final uncached CPU suite, retained semantic/Class-File/member/[operation](../../../../glossary.md#operation)-count/performance
evidence, final [Javadoc](../../../../glossary.md#term-javadoc) and documentation [validation](../../../../glossary.md#term-validation), exact [scope](../../../../glossary.md#term-scope)/[schema](../../../../glossary.md#term-schema)/status checks, independent
clean documentation pass, and completion summary have passed. Any missing gate leaves the task
`In progress`, `Review needed`, `Blocked`, or `Incomplete` with the exact reason.
