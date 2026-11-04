# Task 0007D: Portable Logarithmic, Statistical, and [Norm](../../../../glossary.md#term-norm) Reduction Coverage

## Status

Complete

## Goal

Execute exactly one fully static, resolved-[layout](../../../../glossary.md#layout) `LOG_SUM_EXP`, [`VARIANCE`](../../../../glossary.md#term-variance),
`STANDARD_DEVIATION`, `L1_NORM`, or `L2_NORM` [occurrence](../../../../glossary.md#term-occurrence) on the portable [CPU](../../../../glossary.md#term-cpu) route. Preserve the
[Model](../../../../glossary.md#term-model)'s ordered-[axis](../../../../glossary.md#term-axis) [metadata](../../../../glossary.md#term-metadata), [Shape](../../../../glossary.md#shape) and correction contracts, exact special values, and stable
[finite](../../../../glossary.md#term-finite) [targets](../../../../glossary.md#term-target) through direct generated [bytecode](../../../../glossary.md#term-bytecode) whose hot loops match [operation](../../../../glossary.md#operation)-specific optimal
clean Java implementations.

## [Mental model](../../../../glossary.md#term-mental-model)

```text
one static advanced-reduction node
  -> one shared output-cell geometry
  -> one of three generation-time numerical owners
       log-sum-exp | corrected statistics | norms
  -> one typed generated class
  -> scalar or complete-output-cell parallel-scalar execution
```

The shared geometry decides which logical values belong to an output cell. The selected numerical
[owner](../../../../glossary.md#term-owner) decides how those values are accumulated. [Runtime](../../../../glossary.md#term-runtime) receives only the finalized typed entry and
never interprets [axes](../../../../glossary.md#term-axis), correction, kinds, passes, or numerical policy.

## [Scope](../../../../glossary.md#term-scope)

- Recognize exactly one compiled one-input/one-output [node](../../../../glossary.md#node) whose kind is `LOG_SUM_EXP`,
  [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, or `L2_NORM`.
- Require `MultiAxisReductionAttrs` for [log-sum-exp](../../../../glossary.md#log-sum-exp) and both norms, and
  `StatisticalReductionAttrs` for variance and standard deviation. Consume the already normalized,
  ordered, distinct axis list, <span data-glossary-navigation="true"> ([standard deviation glossary](../../../../glossary.md#term-standard-deviation))</span>`keepDimensions`, and non-negative correction without changing
  Model metadata.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Accept exactly [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16), with identical input/output type and gradient
  eligibility. Reject integral, BOOL, cross-type, wrong-attribute, wrong-arity, or mismatched-output
  occurrences.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [BOOL glossary](../../../../glossary.md#data-type--datatype); [arity glossary](../../../../glossary.md#term-arity))</span>
- Require fully static input/output Shapes and fully resolved non-negative input/output layouts.
  Validate the exact Model-derived retained/removed-axis output Shape and an injective writable
  output. Support empty, partial, and all-axis lists, scalar inputs, zero extents, retained
  Dimensions, arbitrary legal offsets/strides, and zero-stride reads.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [injective glossary](../../../../glossary.md#term-injective); [scalar glossary](../../../../glossary.md#term-scalar); [Dimensions glossary](../../../../glossary.md#dimension); [strides glossary](../../../../glossary.md#term-stride))</span>
- Derive selected-axis membership from the ordered axis list, but traverse each complete selected
  domain in canonical input-axis row-major coordinate order. Axis order remains specialization and
  diagnostic identity but does not alter physical fold order or the abstract result.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Compute checked output-cell and selected-domain counts. Statistical capability requires the
  static selected-domain count `N` to fit a non-negative `long` and satisfy `N > correction` before
  preparation succeeds. Every overflow or invalid denominator fails closed before mutation or
  worker submission.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- Treat an empty [axis](../../../../glossary.md#term-axis) list as a point domain with `N=1` and output Shape equal to the input Shape.
  Emit direct point bodies: raw finite/special identity for log-sum-exp, positive zero for valid
  population variance/standard deviation, and represented absolute value for L1/L2. Correction at
  least one is rejected for that form.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [population variance glossary](../../../../glossary.md#population-variance); [absolute value glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); [L1 glossary](../../../../glossary.md#term-l1); [L2 glossary](../../../../glossary.md#term-l2))</span>
- Treat a selected zero extent as an empty domain: log-sum-exp returns negative infinity; L1/L2
  return positive zero; statistics are ineligible because <span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`N=0` cannot exceed a non-negative
  correction. Empty output Shapes perform no generated call, read, write, worker submission, or
  workspace use.
- Preserve exact special-value priority before [finite](../../../../glossary.md#term-finite) arithmetic:
  - log-sum-exp: any NaN produces the CPU canonical quiet NaN; otherwise any positive infinity
    produces positive infinity; an all-negative-infinity domain or empty domain produces negative
    infinity; other negative infinities contribute zero;<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [CPU glossary](../../../../glossary.md#term-cpu); [quiet NaN glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
  - variance/standard deviation: any NaN or either infinity produces the canonical quiet NaN;
    every valid constant finite domain, including signed zeros, produces positive zero;<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
  - L1/L2: NaN takes priority over infinity; otherwise any infinity produces positive infinity;
    every exact zero result is positive zero.
- Implement finite log-sum-exp with two direct passes: first classify and find the maximum, then
  accumulate `exp(value - maximum)` with a primitive compensated [binary64](../../../../glossary.md#floating-point-precision-and-rounding) sum and finish with
  `maximum + log(sum)`. Preserve a finite singleton's represented value and signed zero directly.
  FLOAT32/BFLOAT16 decode to binary64 intermediates and narrow once at the final store.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- Implement finite variance and standard deviation with a corrected two-pass algorithm. The first
  pass uses the existing exact-sum arithmetic owner to produce the result-type mean. The second
  pass accumulates both deviations and squared deviations with primitive compensated binary64
  sums, applies the correction term <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [mean glossary](../../../../glossary.md#term-mean))</span>`sumSquares - sumDeviation * sumDeviation / N`, clamps only a
  negative roundoff residue to positive zero, divides by `N - correction`, and optionally applies
  the non-negative principal `sqrt`. Generated and clean-Java oracles must use the same ordered
  arithmetic and final narrowing.
- Implement L1 as the exact sum of represented absolute values through the existing exact-sum
  generation seam and one final result-format ties-to-even rounding. Implement L2 with the standard
  scaled sum-of-squares state <span data-glossary-navigation="true"> ([ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [rounding glossary](../../../../glossary.md#term-rounding))</span>`(scale, scaledSquares)` so finite intermediates avoid avoidable
  overflow/underflow, followed by <span data-glossary-navigation="true"> ([underflow glossary](../../../../glossary.md#term-underflow))</span>`scale * sqrt(scaledSquares)` and one final [narrowing](../../../../glossary.md#term-narrowing).
- Use only primitive locals for maxima, compensated sums, means, correction, scale, scaled squares,
  flags, counts, coordinates, and address state. L1 and the statistical first-pass exact mean may
  use the existing alignment-eight <span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>`AGGREGATE_EXACT_STATE` per-range workspace; declare its exact
  maximum required slice once. Log-sum-exp and L2 add no workspace. Add no resource kind.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- Lower flattened output ordinals to complete selected domains. Scalar and parallel-scalar ranges
  own disjoint complete output cells; no selected domain is split, combined, or shared. Multi-pass
  algorithms repeat the same complete domain inside one owning invocation.
- Support matching typed [heap](../../../../glossary.md#term-heap) arrays and native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment) carriers—`double[]`,
  `float[]`, or `short[]`—including every heap/segment/mixed input/output pair. Validate complete
  spans, carrier/access/alignment, output injectivity, input/output overlap, and any workspace
  overlap before the first write or worker submission.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Add one focused advanced-reduction IR and one shared lowering/geometry owner. Add separate
  log-sum-exp, statistical, and norm emitters, plus one focused independent reference owner. Do not
  add the five kinds to ordinary <span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [norm glossary](../../../../glossary.md#term-norm))</span>`CpuAggregateIr` or `CpuAggregateEmitter`.
- Emit one final, field-free, constructor-free generated class with one public static typed
  `invoke` entry. Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly once from 45 to 46; older
  envelopes are incompatible safe misses and historical schema-42 evidence remains unchanged.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Use the corresponding well-written clean Java loop as the generated algorithm, dataflow,
  traversal, pass-count, address-work, Math-call, and performance oracle. Generated bytecode may
  call only the necessary JDK primitive/<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [oracle glossary](../../../../glossary.md#term-test-oracle); [bytecode glossary](../../../../glossary.md#term-bytecode); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`Math`/`MemorySegment` operations and must contain no
  Synaptik runtime numerical helper.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Ordinary, masked, target-Shape, or arg-extrema reductions; softmax/log-softmax; normalization;
  fusion with adjacent nodes; or a decomposition into pointwise and ordinary aggregate nodes.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [fusion glossary](../../../../glossary.md#term-fusion); [nodes glossary](../../../../glossary.md#node); [decomposition glossary](../../../../glossary.md#term-decomposition); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Integral/BOOL execution, FLOAT16, output/accumulator overrides, arbitrary p-norms, weighted or
  multi-output statistics, median, quantiles, covariance, or runtime-selected correction.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16); [accumulator glossary](../../../../glossary.md#accumulator-type); [median glossary](../../../../glossary.md#term-median))</span>
- Dynamic/symbolic Shapes, unresolved layouts, runtime axes, a selected-domain count that does not
  fit <span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [axes glossary](../../../../glossary.md#term-axis))</span>`long`, or deferring `N > correction` [validation](../../../../glossary.md#term-validation) to [Runtime](../../../../glossary.md#term-runtime).
- Partial reductions, splitting one output cell, combine trees, selected-domain vectorization,
  native/vendor routes, materialization, new resource kinds, or Runtime semantic dispatch.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Generated calls to CPU reference implementations or Synaptik arithmetic helpers; allocation,
  boxing, reflection, method handles, `invokedynamic`, generic `Object` descriptors,
  collection/map/string lookup, or avoidable virtual dispatch in generated hot work.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-method-handle-lookup))</span><span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- A universal bitwise mathematical promise for transcendental/statistical finite results. Exact
  special values and generated-versus-direct algorithm identity are required; finite mathematical
  accuracy is validated by the task's explicit tolerance matrix.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [identity glossary](../../../../glossary.md#term-identity-meanings); [tolerance glossary](../../../../glossary.md#term-tolerance); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Public/shared API, Model, Compiler, Training, Runtime, Prepare contract, backend-contract,
  dependency, module boundary, Gradle/toolchain, architecture, conformance, integration, tuning,
  tracing, or persistence-format changes beyond the CPU-private generator schema.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [conformance glossary](../../../../glossary.md#term-conformance); [persistence glossary](../../../../glossary.md#term-persistence); [schema glossary](../../../../glossary.md#term-schema))</span>
- Reopening [CPU](../../../../glossary.md#term-cpu) 0007A1D or creating a detailed task for CPU 0007E or later.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007A1 numerical aggregates`](0007a1-portable-ordinary-numerical-aggregate-reductions.md)
- [`CPU 0007A1A generated self-containment`](0007a1a-generated-scalar-body-self-containment.md)
- [`CPU 0007A1C generated evidence closure`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1H numerical aggregate parity`](0007a1h-numerical-aggregate-residual-parity.md)
- [`CPU 0007C masked reductions`](0007c-portable-masked-reduction-coverage.md)
- [`Model 0018V multi-axis/statistical reductions`](../../../modules/model/tasks/0018v-multi-axis-and-statistical-reductions.md)
- [`Compiler 0005B reduction gradients`](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [`Prepare 0001 resource declaration`](../../../modules/prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md)
- [`Prepare 0002 finalization handoff`](../../../modules/prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Model exclusively owns kinds, attribute pairings, ordered normalized axes, correction, Shape,
  type, gradient eligibility, abstract mathematics, and exact special values. Compiler owns capture,
  revalidation, and gradient construction. CPU implements forward execution only and independently
  fails closed outside this exact subset.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [axes glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- CPU analysis owns capability truth, lowering, algorithm choice, access validation, strategy, and
  exact resources before shared slot assignment. CPU finalization owns generated realization.
  Shared Prepare remains CPU-blind and Runtime invokes only cold-bound typed entries.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Shared selected-domain geometry may be represented once, but the three numerical owners remain
  separate. Ordinary aggregate, masked reduction, and arg-extrema owners keep their existing
  meanings. Existing exact-sum generation remains the sole shared exact arithmetic seam.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([masked reduction glossary](../../../../glossary.md#masked-reduction); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Generated code must preserve the matching optimal clean Java hot-loop and dataflow shape.
  Semantic correctness through a generic helper, bridge, per-element callback, or packed kind
  dispatch is insufficient for a specialized path.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [element glossary](../../../../glossary.md#term-element); [callback glossary](../../../../glossary.md#term-callback); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Any needed public/shared/build/architecture/conformance/integration change, new workspace kind,
  materialization, partial/combine protocol, or ownership uncertainty is a stop condition.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [materialization glossary](../../../../glossary.md#materialization); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Package impact

Existing [CPU](../../../../glossary.md#term-cpu)-private packages changed:

- `io.github.pho001.synaptik.backend.cpu` — exact capability [admission](../../../../glossary.md#term-admission).
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — [immutable](../../../../glossary.md#term-immutable) advanced-reduction [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — shared selected-axis output-cell
  geometry and resource derivation.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — three direct numerical emitters
  and generator dispatch.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent semantics/accuracy
  oracle.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- existing `route.portable`, [`prepare`](../../../../glossary.md#prepare), `executable`, and [`cache`](../../../../glossary.md#term-cache) packages — typed realization,
  declaration, binding, execution, and schema-46 compatibility.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [schema glossary](../../../../glossary.md#term-schema))</span>

Packages added: None.

Type placement:

- Add `CpuAdvancedReductionIr` in `internal.ir` for structural kind/type/axis/retention/correction,
  algorithm version, access, pass, and resource identity.
- Add `CpuAdvancedReductionLowering` in `internal.lowering` for common Shape/layout/output-cell/
  selected-domain geometry and exact workspace derivation.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- Add `CpuLogSumExpEmitter`, `CpuStatisticalReductionEmitter`, and `CpuNormEmitter` in
  `internal.codegen.emit`; each owns only its direct numerical bodies.
- Add `CpuAdvancedReductionReferenceKernel` in `internal.reference`; it independently derives
  coordinates and expected numerical results and is never referenced by generated code.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Add no [facade](../../../../glossary.md#term-facade), manager, [registry](../../../../glossary.md#term-registry), adapter, generic reduction interpreter, or new resource type.

## Affected files

Expected production paths:

- `CpuCapabilityProvider.java` and directly affected [CPU](../../../../glossary.md#term-cpu) package [Javadoc](../../../../glossary.md#term-javadoc);
- `internal/ir/CpuPortableKernelIr.java`, new `CpuAdvancedReductionIr.java`, and [IR](../../../../glossary.md#term-intermediate-representation-ir) package Javadoc;
- `internal/lowering/CpuPartitionLowering.java`, new `CpuAdvancedReductionLowering.java`, and
  lowering package Javadoc;<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new `CpuLogSumExpEmitter.java`,
  `CpuStatisticalReductionEmitter.java`, `CpuNormEmitter.java`, the existing exact-sum generation
  seam where required, and emitter package Javadoc;
- `internal/cache/CpuGeneratorSchema.java` and [cache](../../../../glossary.md#term-cache) package Javadoc;
- `internal/route/portable/CpuPortableRoutePlan.java` and package Javadoc;
- `internal/prepare/CpuPartitionPreparationPlan.java`, `CpuPartitionPreparer.java`,
  `CpuPartitionFinalizer.java`, and package Javadoc;
- `internal/executable/CpuPreparedExecutable.java` and package Javadoc; and
- new `internal/reference/CpuAdvancedReductionReferenceKernel.java`, any minimal existing reference
  dispatch seam, and reference package Javadoc.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>

Expected test paths:

- capability and internal-package inventory tests;
- new matching `CpuAdvancedReductionIrTest`, `CpuAdvancedReductionLoweringTest`,
  `CpuAdvancedReductionGeneratedKernelTest`, and `CpuAdvancedReductionReferenceTest`;
- generated-class shape, specialization, cache, route, prepare/finalization, executable, and
  reference differential tests where their exact inventories or behavior change; and<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization); [prepare glossary](../../../../glossary.md#prepare))</span>
- the stable historical pointwise-ledger test only if schema-current assertions require it; the
  historical ledger resource itself must not change.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [schema glossary](../../../../glossary.md#term-schema))</span>

Expected documentation/[planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  global roadmap. Architecture, Model/API, Compiler, Training, shared Prepare/backend-contract,
  conformance, integration, and build files are review-only unless a stop condition is reached.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

The complete implementation, tests, and documentation may modify or create at most 52 repository
paths: at most 32 production/[Javadoc](../../../../glossary.md#term-javadoc) paths, 15 test paths, and the five named documentation and
[planning](../../../../glossary.md#term-planning) paths. Exactly six new production types are permitted: the [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering), three focused
emitters, and independent reference [owner](../../../../glossary.md#term-owner) listed above.

A 53rd path, a seventh production type, a fourth numerical emitter, any semantic change to
ordinary/masked/arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum) owners, a second exact-sum implementation, a shared/public/build/
architecture/[conformance](../../../../glossary.md#term-conformance)/integration path, [materialization](../../../../glossary.md#materialization), a new resource kind, or partial/combine
state is a stop/replan condition. An omitted expected path does not authorize an unrelated one.

## Acceptance criteria

- Capability supports exactly the five kinds, correct attribute family, all three floating types,
  static resolved Shapes/layouts, exact output metadata, and legal injective output; every other
  occurrence fails closed.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [metadata glossary](../../../../glossary.md#term-metadata); [injective glossary](../../../../glossary.md#term-injective); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Lowering records ordered axes plus selected membership, retention, correction, input/output
  layouts, output count, domain count, algorithm/pass identity, exact-state limb/slice needs, and
  all structural access facts. Checked arithmetic and <span data-glossary-navigation="true"> ([Lowering glossary](../../../../glossary.md#lowering); [identity glossary](../../../../glossary.md#term-identity-meanings); [limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [slice glossary](../../../../glossary.md#slice))</span>`N > correction` validation precede
  declaration and mutation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Differential coverage includes all five kinds and three types; empty/partial/all axes; keep/remove
  Dimensions; scalar and zero extents; first/middle/last/nonadjacent axes; dense/general/zero-stride
  layouts; and heap/segment/mixed carriers.<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension); [scalar glossary](../../../../glossary.md#term-scalar); [stride glossary](../../../../glossary.md#term-stride); [heap glossary](../../../../glossary.md#term-heap))</span>
- Exact special-value tests cover raw NaNs, both infinities, mixed infinity signs, signed zeros,
  subnormals, constant slices, cancellation, finite overflow/underflow, empty domains, point
  domains, and correction values zero, one, <span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [subnormals glossary](../../../../glossary.md#term-subnormal); [constant glossary](../../../../glossary.md#term-constant); [cancellation glossary](../../../../glossary.md#term-cancellation); [finite glossary](../../../../glossary.md#term-finite); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow))</span>`N-1`, `N`, and `Long.MAX_VALUE` where applicable.
- Finite accuracy uses an independent high-precision or StrictMath-based oracle with a frozen test
  tolerance table: L1 must match exact represented rounding; FLOAT64 log-sum-exp/L2/standard
  deviation must be within 8 ulps and variance within 16 ulps; FLOAT32 results must be within
  4 result ulps; BFLOAT16 results must be within one represented BFLOAT16 ulp. Exact special values,
  canonical NaN bits, infinities, and zero signs are raw-bit assertions, not tolerance checks.<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [oracle glossary](../../../../glossary.md#term-test-oracle); [tolerance glossary](../../../../glossary.md#term-tolerance); [L1 glossary](../../../../glossary.md#term-l1); [rounding glossary](../../../../glossary.md#term-rounding); [FLOAT64 glossary](../../../../glossary.md#term-float64); [log-sum-exp glossary](../../../../glossary.md#log-sum-exp); [L2 glossary](../../../../glossary.md#term-l2); [standard deviation glossary](../../../../glossary.md#term-standard-deviation); [variance glossary](../../../../glossary.md#term-variance); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [canonical NaN glossary](../../../../glossary.md#term-canonical-nan))</span>
- Generated and optimal direct Java implementations are raw-bit identical for the same algorithm,
  traversal, carrier/layout, range, and inputs. Statistical direct Java uses the same corrected
  two-pass order; it is separate from the mathematical-accuracy oracle.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- The independent reference derives coordinates and mathematical expectations without production
  lowering, packing, emission, exact-state finalization, or generated execution helpers.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Resource declarations contain exactly one input and one output plus only the existing
  alignment-eight exact-state workspace when required by L1 or statistical mean state. Workspace
  is one disjoint maximum-size slice per simultaneously used output-cell range. There is no
  materialization, partial value, combine state, or hidden scratch.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [mean glossary](../../../../glossary.md#term-mean); [materialization glossary](../../../../glossary.md#materialization))</span>
- Cold execution validates all spans, carrier/access/alignment, output injectivity, input/output
  overlap, and workspace overlap before mutation/submission. Failure leaves input, output, scratch,
  and canaries unchanged and submits no worker work.<span data-glossary-navigation="true"> ([injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Scalar and parallel-scalar results are raw-bit deterministic across ranges, worker counts,
  repeated runs, and concurrent runs because each range owns complete output cells and private
  scratch.
- Generated classes are deterministic, final, field-free, constructor-free, and expose one typed
  static `invoke` with concrete input/output carriers, optional [`MemorySegment`](../../../../glossary.md#term-memorysegment) scratch, packed
  primitive geometry, and `long start/end`. No `Object` or bridge [descriptor](../../../../glossary.md#term-descriptor) is permitted.
- Complete `javap -c -p` and `javap -v -p` inspection proves the selected direct pass/loop shape,
  inlined classification and arithmetic, one final store, and absence of Synaptik numerical calls,
  allocation, boxing, reflection, method handles, `invokedynamic`, collections/maps/strings,
  semantic dispatch, and avoidable coordinate/address recomputation.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- A reviewed generated-member allowlist permits only necessary primitive wrapper raw-bit methods,
  `Math`/`StrictMath` operations selected by the algorithm, and typed `MemorySegment` access. Any
  unexpected member is a failure.
- Specialization/cache identity distinguishes kind, type, ordered axes and selected membership,
  retention, correction, algorithm version/pass shape, access/carrier patterns, domain/resource
  shape, and every emitted-body fact. Runtime bases, offsets, non-code-shaping extents/strides,
  ranges, slots, buffers, workers, and run identity remain cold.<span data-glossary-navigation="true"> ([Specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache); [Runtime glossary](../../../../glossary.md#term-runtime); [buffers glossary](../../../../glossary.md#term-buffer); [run glossary](../../../../glossary.md#term-run))</span>
- Schema advances exactly from 45 to 46. Schema-45 and earlier envelopes safely miss; historical
  schema-42 evidence and resources remain immutable and explicitly historical.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Freeze fifteen dense-array hot cases over [Shape](../../../../glossary.md#shape) `[128, 2048]`, [axes](../../../../glossary.md#term-axis) `[1]`, with all five kinds
  and all three floating types. Use correction one for variance and standard deviation, non-empty
  complete ranges, required preallocated scratch, adversarial-but-finite deterministic values,
  and anti-elimination raw checksums. Add three unchanged-family controls for ordinary SUM, masked
  MEAN, and arg-extrema.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Every generated/direct [target](../../../../glossary.md#term-target) and control passes `<= 1.15x` in each of five isolated forks and
  for the median of fork medians. Forks use identical work and never slow, pad, or route the direct
  oracle through production helpers.<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- Focused tests, one final uncached CPU suite, Javadoc, retained semantics/Class-File/determinism/
  checksum/five-fork evidence, exact scope, and whitespace gates pass. A distinct clean
  documentation context then finalizes Javadocs, CPU guide, glossary impact, planning status, and
  documentation validation without repeating stable Java/performance work.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc); [checksum glossary](../../../../glossary.md#term-checksum); [scope glossary](../../../../glossary.md#term-scope); [planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

The implementation context runs focused tests after stabilization, then one final [CPU](../../../../glossary.md#term-cpu) suite:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAdvancedReductionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAdvancedReductionGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuAdvancedReductionReferenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest
./gradlew :backends:cpu:test --rerun-tasks
```

Retain one [immutable](../../../../glossary.md#term-immutable) evidence bundle outside the repository containing exact source/probe inputs,
environment/[JVM](../../../../glossary.md#term-java-virtual-machine-jvm) facts, all fifteen generated [target](../../../../glossary.md#term-target) classes and three controls, [specialization](../../../../glossary.md#term-specialization)
facts, complete `javap -c -p` and `javap -v -p`, member reports, semantic/accuracy/range/[canary](../../../../glossary.md#term-canary)
results, five isolated `-Xms1g -Xmx1g` [fork](../../../../glossary.md#term-fork-meanings) outputs, rejected whole [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), summaries, inventory,
and a [SHA-256](../../../../glossary.md#term-sha-256) [manifest](../../../../glossary.md#term-manifest). Freeze both direct performance oracles and independent accuracy oracles
before timing. Each fork uses at least five randomized warmups, nine randomized measured rounds,
adaptive [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) of at least 25 ms, deterministic inputs, randomized generated/direct order, and
raw/[checksum](../../../../glossary.md#term-checksum) verification. Reject and retain a whole fork if any target, control, checksum,
semantic, accuracy, or environment gate fails.

The documentation-focused context receives the stabilized diff, final CPU XML, and retained
evidence. It reuses successful Java/performance evidence unless executable behavior changes or a
concrete stale-evidence risk is recorded. After final Javadocs it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also inspects rendered Javadocs and validates Markdown links/anchors/fences/newlines/whitespace,
exact path/type ceilings, [schema](../../../../glossary.md#term-schema) 46 and historical schema-42 wording, CPU 0007A2/0007B/0007C/
0007D Complete only after all gates, CPU 0007A1D Review needed, CPU 0007E as the next Draft task
without a detailed specification, and empty staging. Repository-wide, architecture,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration suites remain deferred to CPU 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) because no shared
boundary changes.

## Dependencies

- Complete Model 0018V owns all five kinds, attribute pairings, ordered normalized axes,
  correction, Shape/type metadata, abstract finite targets, special values, and empty/point-domain
  meaning.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [axes glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [finite glossary](../../../../glossary.md#term-finite); [targets glossary](../../../../glossary.md#term-target))</span>
- Complete Compiler 0005B and current inference own capture, static/dynamic validation, and
  gradient construction; this task changes forward CPU execution only.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [gradient glossary](../../../../glossary.md#term-gradient); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Complete CPU 0007A1 supplies exact floating sum state, represented rounding, workspace shape,
  complete-output-cell execution, and ordinary multi-axis geometry precedent.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding))</span>
- Complete CPU 0007A1A/0007A1C/0007A1H supply generated self-containment, Class-File/member
  inspection, clean-Java equivalence, and five-fork performance gates. CPU 0007A1D remains
  historical Review needed and is not a dependency.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Complete CPU 0007A2/0007B/0007C supply the current schema-45 preparation, finalization, cache,
  focused-owner, mixed-carrier, and deterministic parallel frontier.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [owner glossary](../../../../glossary.md#term-owner); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Existing shared Prepare/Runtime resource contracts can carry the two buffers and existing exact
  state without modification.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [carry glossary](../../../../glossary.md#term-carry); [buffers glossary](../../../../glossary.md#term-buffer))</span>

## Follow-up tasks

- CPU 0007E remains the next ordered Draft task for stable softmax/log-softmax and receives no
  detailed specification here.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- CPU 0007F and later work remain Draft in existing order. CPU 0009 or CI retains repository-wide
  capability, conformance, and integration closure.<span data-glossary-navigation="true"> ([CI glossary](../../../../glossary.md#term-continuous-integration-ci); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- CPU 0007A1D remains historical Review needed; resolving its incomplete performance evidence is
  independent and not a prerequisite for this family.

## Architecture impact

Expected impact: None.

This task adds one [CPU](../../../../glossary.md#term-cpu)-private generated execution family behind existing [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler),
[Prepare](../../../../glossary.md#prepare), and [Runtime](../../../../glossary.md#term-runtime) contracts. If implementation reveals a need for a public/shared contract,
[dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module), resource kind, architecture rule, or build change, stop and report rather than
editing that boundary.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007D. Work on the existing worktree
without committing, pushing, staging, resetting, reverting, deleting, or modifying unrelated
work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, CPU task 0007D and
its dependencies, Model 0018V, Compiler 0005B and current verification/gradient code, current CPU
capability/IR/lowering/emitter/reference/route/prepare/finalization/executable/cache source and
tests, documentation rules, and the General/API-Javadoc/Backend-guide/Planning profiles.

Implement exactly the Ready specification. Preserve ordinary, masked, and arg-extrema owners; add
the six focused advanced-reduction types; advance schema 45 to 46; and satisfy every semantic,
accuracy, resource, generated-Class-File, determinism, checksum, control, and five-fork
optimal-direct-Java gate without slowing the direct oracles.

Stop on any task ceiling, architecture uncertainty, or scope conflict. Hand the stabilized diff,
CPU XML, and retained evidence to a distinct clean documentation-focused context to finalize
documentation and planning in the same change. Do not mark Complete until all gates pass. Do not
commit, push, or stage.
```

## Local decisions

- One shared advanced-reduction IR/lowering owns static output-cell geometry because all five
  kinds share exactly one input, one output, ordered selected axes, retention, and complete-cell
  parallel ownership. Three emitters retain distinct numerical responsibilities and avoid a god
  emitter or runtime kind switch.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [axes glossary](../../../../glossary.md#term-axis); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Log-sum-exp uses max-shift plus compensated binary64 accumulation. The singleton direct path is
  required to preserve represented finite signed zero exactly.<span data-glossary-navigation="true"> ([Log-sum-exp glossary](../../../../glossary.md#log-sum-exp); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [singleton glossary](../../../../glossary.md#term-singleton); [finite glossary](../../../../glossary.md#term-finite); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- Corrected two-pass statistics reuse exact-sum generation for the first-pass mean, then compensate
  second-pass deviations and squared deviations. The correction term repairs use of the rounded
  stored mean without introducing another exact arithmetic implementation.<span data-glossary-navigation="true"> ([mean glossary](../../../../glossary.md#term-mean))</span>
- L1 reuses exact-sum state over absolute represented values. L2 uses scaled sum-of-squares rather
  than naively squaring first, avoiding avoidable finite overflow and underflow.<span data-glossary-navigation="true"> ([L1 glossary](../../../../glossary.md#term-l1); [L2 glossary](../../../../glossary.md#term-l2); [overflow and underflow glossary](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow))</span>
- FLOAT32 and BFLOAT16 may use binary64 intermediates because Model allows equal-or-wider
  accumulation; output type remains unchanged and narrowing occurs once. This is not a public
  promotion or cross-backend bitwise promise.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [Model glossary](../../../../glossary.md#term-model); [narrowing glossary](../../../../glossary.md#term-narrowing); [backend glossary](../../../../glossary.md#backend))</span>
- Statistics need exact-state scratch for their first pass and L1 needs it for exact absolute sum.
  The other state is primitive-local. One maximum per-range slice keeps declarations truthful
  without a new workspace kind.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- No selected domain is split. This gives deterministic scalar/parallel parity and avoids a
  combine algorithm before a later task explicitly proves one.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Schema 46 records the new family, algorithm/pass identities, correction, ordered axes, and
  resource shapes. Runtime-only geometry remains cold where it does not change generated bytes.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [shapes glossary](../../../../glossary.md#shape))</span>

## Known limitations

- Only fully static resolved-layout scalar/parallel-scalar execution is supported. Dynamic Shapes,
  selected-domain vectorization, partial/combine execution, fusion, and native routes remain
  unsupported.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [Shapes glossary](../../../../glossary.md#shape); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Finite transcendental/statistical results follow the selected stable CPU algorithms and test
  tolerances; only special-value and generated/direct algorithm results are raw-bit contracts.<span data-glossary-navigation="true"> ([Finite glossary](../../../../glossary.md#term-finite); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Large multi-pass reductions reread the input. Materialization and autotuned alternative
  algorithms are deferred to their existing later owners.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Materialization glossary](../../../../glossary.md#materialization))</span>
- The performance gate applies only to its frozen cases, environment, and protocol; it is not a
  universal performance or tuning claim.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation and review produced the final uncached [CPU](../../../../glossary.md#term-cpu) report with 416 tests, one expected
opt-in [persistence](../../../../glossary.md#term-persistence) skip, zero failures, and zero errors. The implementation context also completed
the focused advanced-reduction suites. This distinct documentation-focused context reused that
stable Java evidence because it changed no executable Java behavior; final CPU [Javadoc](../../../../glossary.md#term-javadoc) generation
and all documentation gates passed after the documentation edits.

The [immutable](../../../../glossary.md#term-immutable) evidence bundle at `/private/tmp/synaptik-cpu-0007d-evidence` contains fifteen
generated [targets](../../../../glossary.md#term-target) (five kinds across [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), and [BFLOAT16](../../../../glossary.md#term-bfloat16)) and three unchanged-family
controls. It retains complete semantics, [descriptor](../../../../glossary.md#term-descriptor), compatibility, Class-File, `javap -c -p`,
`javap -v -p`, member, [checksum](../../../../glossary.md#term-checksum), environment, and five-isolated-[fork](../../../../glossary.md#term-fork-meanings) evidence. All fifteen targets
and three controls passed the `<= 1.15x` fork and aggregate gates. The worst fork ratio was
`1.099406784x` and the worst aggregate ratio was `1.087934030x`, both FLOAT64 `LOG_SUM_EXP`.
The reviewed member report contains no forbidden hit. The [SHA-256](../../../../glossary.md#term-sha-256) [manifest](../../../../glossary.md#term-manifest) validates every
retained [artifact](../../../../glossary.md#term-artifact).

Final [scope](../../../../glossary.md#term-scope) is exactly 45 repository paths: 27 production/Javadoc paths, 13 test paths, and the
five named documentation/[planning](../../../../glossary.md#term-planning) paths. Exactly six production types are new: one [IR](../../../../glossary.md#term-intermediate-representation-ir), one
lowerer, three focused numerical emitters, and one independent reference [owner](../../../../glossary.md#term-owner). Generator [schema](../../../../glossary.md#term-schema)
is exactly 46; schema-45 and earlier artifacts are safe misses, while retained schema-42 evidence
remains explicitly historical. Staging is empty.

Documentation validation covered the General, [API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc, [Backend](../../../../glossary.md#backend) Guide, Planning, and Example
profiles; local Markdown targets and heading anchors; fenced blocks; terminology and glossary
[alignment](../../../../glossary.md#term-alignment); status [synchronization](../../../../glossary.md#term-synchronization); manual rendering of the changed guide, glossary, and planning
sections; newline and whitespace checks; exact scope/type counts; CPU Javadoc rendering; and final
diff review. CPU 0007A2, 0007B, 0007C, and 0007D are Complete; CPU 0007A1D remains historical
Review needed; CPU 0007E remains the next Draft task without a detailed specification.

## Implementation notes

The implementation added shared selected-domain geometry while keeping the numerical [owners](../../../../glossary.md#term-owner)
separate. `CpuLogSumExpEmitter`, `CpuStatisticalReductionEmitter`, and `CpuNormEmitter` emit their
own numerical bodies; the [norm](../../../../glossary.md#term-norm) emitter shares only primitive [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/[carrier](../../../../glossary.md#term-storage-carrier), classification,
compensated-add, [narrowing](../../../../glossary.md#term-narrowing), and store mechanics. Generated [heap](../../../../glossary.md#term-heap) and segment entries preserve the
matching optimal clean Java pass, traversal, [dataflow](../../../../glossary.md#term-dataflow), and avoidable-overhead [shape](../../../../glossary.md#shape).

No architecture or shared contract changed. Review found no required change to
`ARCHITECTURE.md`, current architecture explanations, public [Model](../../../../glossary.md#term-model) or [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), [Compiler](../../../../glossary.md#term-compiler),
Training, [Runtime](../../../../glossary.md#term-runtime) semantics, shared [Prepare](../../../../glossary.md#prepare), [backend](../../../../glossary.md#backend)-contract, architecture tests,
backend-[conformance](../../../../glossary.md#term-conformance) tests, integration tests, [Gradle](../../../../glossary.md#term-gradle)/[toolchain](../../../../glossary.md#term-toolchain) configuration, or unrelated
[modules](../../../../glossary.md#term-module). The reason is that the new family is [CPU](../../../../glossary.md#term-cpu)-private forward realization behind existing
[operation](../../../../glossary.md#operation), preparation-resource, finalization, and cold-bound execution contracts. It adds no
[dependency](../../../../glossary.md#term-dependency), module boundary, resource kind, public vocabulary, [gradient](../../../../glossary.md#term-gradient) rule, or cross-backend
promise.

## Completion summary

Completed [CPU](../../../../glossary.md#term-cpu)-private generated execution for fully static resolved-[layout](../../../../glossary.md#layout) `LOG_SUM_EXP`,
[`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM` over [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), and [BFLOAT16](../../../../glossary.md#term-bfloat16).
The implementation preserves ordered [axes](../../../../glossary.md#term-axis), correction, empty/point-domain and special-value
rules, complete-output-cell deterministic execution, truthful exact-state resources, typed [heap](../../../../glossary.md#term-heap)/
segment/mixed carriers, three focused numerical emitters, and [schema](../../../../glossary.md#term-schema) 46. Documentation and
[planning](../../../../glossary.md#term-planning) now describe the implemented boundary and retained evidence without promising public,
[compiler](../../../../glossary.md#term-compiler)/training, [vector](../../../../glossary.md#term-vector)/native/[fusion](../../../../glossary.md#term-fusion)/[materialization](../../../../glossary.md#materialization)/dynamic-[Shape](../../../../glossary.md#shape), partial/combine, or
cross-[backend](../../../../glossary.md#backend) behavior.

Files changed or created are the 45 paths recorded by the final diff: 27 production/[Javadoc](../../../../glossary.md#term-javadoc) paths,
13 test paths, and `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master
plan, and the global roadmap. [Validation](../../../../glossary.md#term-validation) passed as recorded above. There are no unresolved issues
or required follow-up for CPU 0007D. CPU 0007E remains the next ordered Draft task.

Status: Complete
