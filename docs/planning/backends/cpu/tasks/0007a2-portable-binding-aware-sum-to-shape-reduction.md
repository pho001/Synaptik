# Task 0007A2: Portable [Binding](../../../../glossary.md#term-binding)-Aware [Sum-to-Shape](../../../../glossary.md#sum-to-shape) Reduction

## Status

Complete

## Goal

Implement portable generated [CPU](../../../../glossary.md#term-cpu) execution for exactly one fully bound, fully static,
resolved-[layout](../../../../glossary.md#layout) [Model](../../../../glossary.md#term-model) `AggregateReductionKind.SUM` [occurrence](../../../../glossary.md#term-occurrence) carrying exact
`SumToShapeAttrs`. CPU analysis must prove the Model's right-aligned [target](../../../../glossary.md#term-target)-one-or-source-equal
obligation, derive leading and aligned reduction geometry, and execute all five current numeric
types through direct generated [scalar](../../../../glossary.md#term-scalar) or parallel-scalar hot loops.

The implementation reuses CPU 0007A1's exact floating SUM state, one-result-format [rounding](../../../../glossary.md#term-rounding),
integral modular addition, output-cell [ownership](../../../../glossary.md#term-ownership-meanings), workspace [lifecycle](../../../../glossary.md#lifecycle), and direct optimal Java
[oracle](../../../../glossary.md#term-test-oracle). It adds no dynamic-[Shape](../../../../glossary.md#shape) execution or second reduction architecture.

```text
bound source Shape + exact bound target Shape
  -> prove target rank <= source rank and every aligned pair is equal or target one
  -> reduce every leading source axis
  -> preserve equal aligned axes; reduce and retain target-one aligned axes
  -> direct generated copy when no axis is reduced, otherwise direct generated SUM
```

## [Scope](../../../../glossary.md#term-scope)

### Exact [Model](../../../../glossary.md#term-model) form and bound [validation](../../../../glossary.md#term-validation)

- Admit only `AggregateReductionKind.SUM` with exactly one input, one output, and
  `SumToShapeAttrs`.
- Admit [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), and [`INT64`](../../../../glossary.md#data-type--datatype); reject [`BOOL`](../../../../glossary.md#data-type--datatype) and every type,
  kind, arity, or attributes combination outside this matrix before route selection.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Require fully static input, target, and output Shapes and resolved input/output layouts. This is
  the existing Prepare boundary: unresolved Model Dimensions must already have been bound and
  proved before CPU analysis. A dynamic or symbolic descriptor reaching CPU fails closed; CPU,
  generated code, and Runtime do not bind Dimensions.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [layouts glossary](../../../../glossary.md#layout); [Prepare glossary](../../../../glossary.md#prepare); [Dimensions glossary](../../../../glossary.md#dimension); [CPU glossary](../../../../glossary.md#term-cpu); [descriptor glossary](../../../../glossary.md#term-descriptor); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Require the output [Shape](../../../../glossary.md#shape) to equal the exact bound `SumToShapeAttrs.targetShape()` and retain the
  input data type. Require target rank not to exceed input rank.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [rank glossary](../../../../glossary.md#term-rank))</span>
- Let `leading = inputRank - targetRank`. Every input [axis](../../../../glossary.md#term-axis) below `leading` is reduced and removed.
  For target axis `t`, aligned input axis `i = leading + t` is handled in this exact order:
  equal bound extents preserve the coordinate without reduction; otherwise target extent one
  reduces input axis <span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [extent glossary](../../../../glossary.md#term-extent))</span>`i` and retains the target position; every other pair is rejected before
  finalization or execution.
- Canonical selected-axis membership is increasing input-axis order: all leading axes followed by
  aligned target-one axes whose source extent differs from one. Equality takes precedence, so an
  aligned source-one/target-one pair is preserved rather than treated as an arithmetic reduction.<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Equal bound input and target Shapes therefore select no reduction axes but still describe one
  explicit SUM occurrence with a distinct materialized output.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>

### Leading and aligned geometry

- Extend the existing `CpuAggregateIr`/`CpuAggregateLowering` family in place with a closed
  `SUM_TO_SHAPE` form and exact [target](../../../../glossary.md#term-target)-[alignment](../../../../glossary.md#term-alignment) facts. Do not create `CpuSumToShapeIr`, a second
  access planner, a generic reduction plan, or a runtime binding abstraction.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding))</span>
- Record enough cold geometry to map each flattened target coordinate to its right-aligned source
  coordinates and to visit the Cartesian product of selected leading/aligned axes in canonical
  row-major input-coordinate order. The mapping must not depend on <span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Cartesian product glossary](../../../../glossary.md#term-cartesian-product); [axes glossary](../../../../glossary.md#term-axis); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major))</span>`keepDimensions`, which is not
  a `SumToShapeAttrs` property.
- `outputCount` is the exact target [element](../../../../glossary.md#term-element) count. `domainCount` is the checked product of the
  selected source extents, or one when no axis is selected. Geometry and byte arithmetic use
  checked `long` operations and fail deterministically during capability/[lowering](../../../../glossary.md#lowering)/preparation.
- A selected zero [extent](../../../../glossary.md#term-extent) makes `domainCount == 0`; every existing target coordinate receives the
  SUM empty identity. An equal aligned zero/zero pair is preserved and produces zero output
  coordinates. A zero extent on an unselected axis likewise makes <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`outputCount == 0`, so no write,
  workspace allocation, generated call, or worker submission occurs.
- Scalar input accepts only scalar target and follows the no-reduction copy path. Scalar target
  reduces every axis of a non-scalar input.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar))</span>
- Support every currently legal aggregate input/output layout and carrier combination: exact
  typed heap arrays, <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier); [heap glossary](../../../../glossary.md#term-heap))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment), and mixed input/output carriers; nonzero offsets;
  non-negative positive or zero read strides; arbitrary injective output layouts; and arbitrary
  valid half-open output-cell ranges. Do not narrow the established resolved-layout contract.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective); [half-open glossary](../../../../glossary.md#term-half-open-interval))</span>

### Numerical, copy, and deterministic execution rules

- When at least one [axis](../../../../glossary.md#term-axis) is actually reduced, inherit [CPU](../../../../glossary.md#term-cpu) 0007A1 SUM exactly:
  - floating finite values use the fixed-width signed superaccumulator in least-subnormal units
    and one round-to-nearest, ties-to-even conversion to the result type;<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [superaccumulator glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [subnormal glossary](../../../../glossary.md#term-subnormal); [round-to-nearest, ties-to-even glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
  - floating NaN, infinity, signed-zero, subnormal, overflow, underflow, and empty-domain behavior
    remains the schema-29 ordinary SUM contract;<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [schema glossary](../../../../glossary.md#term-schema))</span>
  - [`INT32`](../../../../glossary.md#data-type--datatype) and [`INT64`](../../../../glossary.md#data-type--datatype) use same-width modular addition with zero [identity](../../../../glossary.md#term-identity-meanings).
- Reuse `CpuExactSumEmitter` as the sole generation-time exact floating accumulation/conversion
  owner. Do not duplicate its algorithm or add a runtime numerical helper.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- When no axis is reduced, copy the corresponding represented input value exactly. This includes
  raw floating NaN payload/sign/signaling bits and signed zero; it must not pass one value through
  SUM classification or rounding. The direct generated copy is still part of the aggregate
  artifact and must not call the affine-copy bridge or another prepared executable.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed zero glossary](../../../../glossary.md#term-signed-zero); [rounding glossary](../../../../glossary.md#term-rounding); [artifact glossary](../../../../glossary.md#term-artifact); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- Scalar and parallel-scalar execution partition only disjoint complete output-cell ranges. One
  worker owns the complete reduction domain and all mutable state for every output cell it writes.
  Never split or merge one reduction domain. Results are bit-identical across worker counts,
  chunking, repeated invocation, and concurrent reuse of one immutable prepared executable with
  distinct <span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [partition glossary](../../../../glossary.md#partition); [mutable glossary](../../../../glossary.md#term-mutable); [immutable glossary](../../../../glossary.md#term-immutable))</span>[`RunState`](../../../../glossary.md#run-state--runstate) instances.
- Generated dense heap forms use cold-proved primitive integer loop/address state where safe;
  every other supported layout/carrier/range retains a direct typed long-address fallback. Both
  must preserve the clean optimal Java algorithm and dataflow shape.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier); [dataflow glossary](../../../../glossary.md#term-dataflow); [shape glossary](../../../../glossary.md#shape))</span>

### Overlap, resources, and [lifecycle](../../../../glossary.md#lifecycle)

- Declare exactly one distinct input buffer and one distinct injective output buffer. Declare no
  materialization.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [injective glossary](../../../../glossary.md#term-injective); [materialization glossary](../../../../glossary.md#materialization))</span>
- Reject complete physical input/output overlap before any output mutation, including equal-Shape
  copy cases. Preserve carrier compatibility, liveness, writability, referenced-span, alignment,
  output-injectivity, and arbitrary-range validation at the established cold binding boundary.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [carrier glossary](../../../../glossary.md#term-storage-carrier); [liveness glossary](../../../../glossary.md#term-liveness); [alignment glossary](../../../../glossary.md#term-alignment); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding))</span>
- Floating rows with at least one selected reduction [axis](../../../../glossary.md#term-axis) reuse workspace purpose
  `AGGREGATE_EXACT_STATE`, requirement ID zero, alignment eight, and the exact CPU 0007A1 SUM
  slice layout <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [slice glossary](../../../../glossary.md#slice); [layout glossary](../../../../glossary.md#layout))</span>`8 + 8 * sumLimbCount` bytes. [Limb](../../../../glossary.md#exact-superaccumulator-and-limbs) count is derived from the bound `domainCount`.
  Analysis declares one disjoint slice per simultaneously executable output range before shared
  slot assignment and enforces `maximumAdditionalBytes` exactly.
- Integral reductions, all no-reduction represented-bit copies, and every zero-output occurrence
  declare zero workspace. A selected empty floating domain with positive output count still needs
  exact-state slices because the generated SUM path owns initialization and finalization.<span data-glossary-navigation="true"> ([represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- The immutable prepared executable captures geometry, specialization, assigned slot identities,
  artifact, handle, and worker policy only. It captures no writable accumulator, Arena, carrier,
  address, or run-owned workspace. Runtime continues to own allocation, isolation, cleanup, and
  publication.<span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings); [identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [accumulator glossary](../../../../glossary.md#accumulator-type); [Arena glossary](../../../../glossary.md#term-arena); [run glossary](../../../../glossary.md#term-run); [Runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication))</span>
- Complete buffer and scratch presence/size/alignment/access/physical-overlap validation before
  the first write. Failure mutates no output; preparation failure publishes no executable.

### Generated [hot path](../../../../glossary.md#term-hot-path) and [schema](../../../../glossary.md#term-schema) 43

- Extend the existing aggregate generated entry [shapes](../../../../glossary.md#shape):
  - reduction, integral: `(inputCarrier, outputCarrier, long[] geometry, long start, long end)void`;
  - reduction, floating: `(inputCarrier, outputCarrier, MemorySegment scratch, long[] geometry,
    long start, long end)void`;
  - no-reduction copy, all five types: the workspace-free integral-shaped entry.
- Input and output carriers are independently the exact matching `[D`, `[F`, `[S`, `[I`, `[J`,
  or [`MemorySegment`](../../../../glossary.md#term-memorysegment) form. No entry admits `Object`.
- Generated loops directly contain target-coordinate mapping, selected-axis traversal, typed
  loads/stores, modular addition or exact-sum limb work, and final conversion. Hot code contains no
  Model/Compiler/Runtime object, <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [axis glossary](../../../../glossary.md#term-axis); [limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime))</span>[`Operation`](../../../../glossary.md#operation), attributes, [graph](../../../../glossary.md#graph) inspection, bridge to
  `CpuScalarReferenceKernel`, `CpuExactSumEmitter`, affine copy, method handle, collection,
  boxing, reflection, allocation, string/map lookup, route/type/form/carrier/layout switch,
  per-element virtual dispatch, or avoidable division/modulo.<span data-glossary-navigation="true"> ([method handle glossary](../../../../glossary.md#term-method-handle); [lookup glossary](../../../../glossary.md#term-method-handle-lookup); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 42 to 43. This task changes generated bytes by
  adding the SUM-to-Shape form, target-alignment mapping, selected-axis traversal, and exact
  represented-copy body, and adds new form/mapping/resource facts to structural compatibility.
  Current-only compatibility therefore treats every schema-42 envelope as an incompatible safe
  miss; no migration reader or partial reuse is added. Existing schema-42 specializations must
  regenerate with unchanged semantics, and representative pre-A2 operations must be verified
  against preserved schema-42 bytes to detect unintended code-shape drift. The A1O ledger
  resource remains immutable historical evidence with <span data-glossary-navigation="true"> ([SUM-to-Shape glossary](../../../../glossary.md#sum-to-shape); [alignment glossary](../../../../glossary.md#term-alignment); [immutable glossary](../../../../glossary.md#term-immutable))</span>`generated-schema=42`; its test must prove
  that this historical metadata is compatible evidence while independently requiring the current<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
  `CpuGeneratorSchema.CURRENT_VERSION` to be exactly 43.
- Include form, ranks/alignment, selected-axis membership, reduced-versus-copy mode, exact-state
  shape, access regimes, carriers, and every other byte/resource-changing fact in IR,
  specialization, entry-descriptor, manifest, and cache compatibility. Concrete carriers,
  offsets, addresses, slots, ranges, worker count/chunk size, graph identities, and run identity
  remain cold when they do not change emitted code.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [specialization glossary](../../../../glossary.md#term-specialization); [descriptor glossary](../../../../glossary.md#term-descriptor); [manifest glossary](../../../../glossary.md#term-manifest); [cache glossary](../../../../glossary.md#term-cache); [run glossary](../../../../glossary.md#term-run); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

### Correctness, code-[shape](../../../../glossary.md#shape), and performance evidence

- Add an independent test-only oracle that computes the right-aligned mapping itself. Reduced
  floating cases use the existing independent <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>`BigInteger`/rational SUM oracle without sharing
  emitter logic; no-reduction cases compare raw represented bits directly.<span data-glossary-navigation="true"> ([represented bits glossary](../../../../glossary.md#term-represented-bits))</span>
- Cover all five types; scalar and lower-rank targets; leading-only, aligned-only, combined
  leading/aligned, equal-Shape/no-reduction, selected/unselected zero extents; dense/general
  layouts; offsets; positive and zero read strides; injective strided output; heap/segment/mixed
  carriers; partial/empty ranges; scalar/parallel worker counts; repeated/concurrent reuse;
  overlap/resource failures; cache hit/miss/corruption; and schema-42 safe misses.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [layouts glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective); [heap glossary](../../../../glossary.md#term-heap); [cache hit glossary](../../../../glossary.md#cache-and-memoization); [schema glossary](../../../../glossary.md#term-schema))</span>
- Floating adversarial vectors cover cancellation, both halfway parities, finite overflow,
  normal/subnormal boundaries, signed underflow, smallest subnormals, NaN classes, infinities,
  every zero-sign case, and empty domains. Integral vectors cover modular wrap and empty domains.<span data-glossary-navigation="true"> ([vectors glossary](../../../../glossary.md#term-vector); [cancellation glossary](../../../../glossary.md#term-cancellation); [finite glossary](../../../../glossary.md#term-finite); [overflow glossary](../../../../glossary.md#term-overflow); [subnormal glossary](../../../../glossary.md#term-subnormal); [underflow glossary](../../../../glossary.md#term-underflow); [NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Retain Class-File and `javap -c/-v` evidence for at least FLOAT64 combined leading/aligned
  reduction, BFLOAT16 general mixed-carrier reduction, INT64 aligned reduction, and FLOAT32
  equal-Shape copy. Assert exact descriptors, deterministic bytes, allowed member references,
  direct loop ownership, schema-43 compatibility, and forbidden helper/object leakage.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [carrier glossary](../../../../glossary.md#term-storage-carrier); [INT64 glossary](../../../../glossary.md#data-type--datatype); [FLOAT32 glossary](../../../../glossary.md#term-float32); [descriptors glossary](../../../../glossary.md#term-descriptor); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Use a clean direct primitive Java baseline with the same mapping, exact floating state layout,
  reset, decoding, rounding, carrier work, and output-cell traversal. Measure these six dense
  heap-array cases on source <span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding))</span>`[64, 128, 256]` to [target](../../../../glossary.md#term-target) `[128, 1]`: reduced SUM for all five
  numeric types, plus FLOAT32 equal-Shape represented copy on `[1,048,576]`. Verify raw bits before
  timing. Run five isolated JVM forks, five warmup rounds, nine randomized measured rounds, adaptive
  batches of at least 25 ms, and <span data-glossary-navigation="true"> ([JVM forks glossary](../../../../glossary.md#java-preview-incubator-modules-and-compiler-tiers))</span><span data-glossary-navigation="true"> ([raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [Run glossary](../../../../glossary.md#term-run); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [warmup glossary](../../../../glossary.md#term-warmup); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>`-Xms1g -Xmx1g`. Every fork and median-of-fork-medians
  generated/direct ratio must be <span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x`. This evidence is a [release](../../../../glossary.md#term-release) gate, never a tuning input.

## Out of [scope](../../../../glossary.md#term-scope)

- Unresolved/dynamic Shape execution, a shared binding API, runtime Dimension values, or late
  route/resource selection.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [runtime glossary](../../../../glossary.md#term-runtime); [Dimension glossary](../../../../glossary.md#dimension))</span>
- Ordinary full/single-/multi-axis or masked SUM changes; MEAN/PROD target-Shape forms; BOOL;
  arg-extrema; later reduction, softmax, or normalization families.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [MEAN glossary](../../../../glossary.md#term-mean); [target glossary](../../../../glossary.md#term-target); [BOOL glossary](../../../../glossary.md#data-type--datatype); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Fusion, vector reduction, domain splitting/combination, native/OpenBLAS/vendor routes,
  measurement-driven preparation, tuning, or cache mutation outside existing cold artifact reuse.<span data-glossary-navigation="true"> ([Fusion glossary](../../../../glossary.md#term-fusion); [vector glossary](../../../../glossary.md#term-vector); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [cache glossary](../../../../glossary.md#term-cache); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- New Model/Compiler/Planning/Prepare/Runtime/Engine APIs, semantic changes, dependencies, Gradle
  changes, architecture contracts/ADRs/tests, another backend, conformance, or integration work.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A second IR/lowering/emitter/resource hierarchy, generic helper/manager/service, public CPU API,
  or unrelated refactoring.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering))</span>
- Repair or reinterpretation of [CPU](../../../../glossary.md#term-cpu) 0007A1D's historical `Review needed` result.
- Editing or regenerating A1O's `operation-family-form-ledger-v2.tsv`; its
  `generated-schema=42` metadata records historical generated evidence and is not a current-schema
  assertion.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [schema glossary](../../../../glossary.md#term-schema))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current-architecture-plan.md`](../../../../architecture/current-architecture-plan.md)
- [`performance-evidence-and-tuning.md`](../../../../architecture/performance-evidence-and-tuning.md)
- [`runtime-prepare-backend-boundary.md`](../../../../architecture/runtime-prepare-backend-boundary.md)
- [`planning-guide.md`](../../../planning-guide.md)
- [`documentation-rules.md`](../../../../developer-guide/documentation-rules.md)
- [`general documentation profile`](../../../../developer-guide/documentation/general-style.md)
- [`backend-guide documentation profile`](../../../../developer-guide/documentation/backend-guide-style.md)
- [`API/Javadoc documentation profile`](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [`planning documentation profile`](../../../../developer-guide/documentation/planning-style.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0005B`](0005b-universal-access-plans-and-right-aligned-broadcasting.md)
- [`CPU 0007A`](0007a-portable-ordinary-extrema-and-boolean-reductions.md)
- [`CPU 0007A0`](0007a0-generated-hot-path-parity-correction.md)
- [`CPU 0007A1`](0007a1-portable-ordinary-numerical-aggregate-reductions.md)
- [`CPU 0007A1A`](0007a1a-generated-scalar-body-self-containment.md)
- [`CPU 0007A1H`](0007a1h-numerical-aggregate-residual-parity.md)
- [`CPU 0007A1N`](0007a1n-multi-axis-min-residual-parity.md)
- [`CPU 0007A1O`](0007a1o-pointwise-ledger-evidence-reconciliation.md)
- [`Model 0023A`](../../../modules/model/tasks/0023a-binding-aware-sum-to-shape.md)
- [`Compiler 0005B`](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)

## Architecture constraints

- [Model](../../../../glossary.md#term-model) owns SUM/`SumToShapeAttrs` meaning, type and Shape semantics. Compiler owns retained
  binding constraints and proof. CPU consumes only the fully resolved occurrence during backend
  analysis and owns lowering, route choice, exact resources, specialization, generation, and
  execution.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding); [CPU glossary](../../../../glossary.md#term-cpu); [occurrence glossary](../../../../glossary.md#term-occurrence); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Shared Prepare remains CPU-blind: CPU analysis declares every buffer/workspace requirement
  before assignment; finalization resolves assignments and constructs the artifact without
  changing route or resources.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [buffer glossary](../../../../glossary.md#term-buffer); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Runtime executes one immutable prepared recipe with isolated per-run resources and performs no
  graph inspection, semantic dispatch, route selection, generation, or cache work.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable); [run glossary](../../../../glossary.md#term-run); [graph glossary](../../../../glossary.md#graph); [dispatch glossary](../../../../glossary.md#term-dispatch); [cache glossary](../../../../glossary.md#term-cache))</span>
- Direct generated bytecode must preserve the optimal clean Java semantic algorithm, hot-loop and
  dataflow shape, and avoidable-overhead profile. Any deviation requires a recorded technical
  reason and evidence.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>
- Preserve existing public API, module dependencies, and architecture. If a shared contract or
  architecture change proves necessary, stop before editing and request an explicit decision.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used or changed:

- `io.github.pho001.synaptik.backend.cpu` — truthful [occurrence](../../../../glossary.md#term-occurrence)-local capability and package status.
- `internal.ir` — existing aggregate [IR](../../../../glossary.md#term-intermediate-representation-ir) gains the closed [target](../../../../glossary.md#term-target)-[Shape](../../../../glossary.md#shape) form and mapping [identity](../../../../glossary.md#term-identity-meanings).
- `internal.lowering` — existing aggregate lowerer owns bound [validation](../../../../glossary.md#term-validation) and geometry derivation.
- `internal.codegen.emit` — existing aggregate emitter owns direct mapping/copy/SUM bodies and
  reuses the existing exact-sum generation owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `internal.cache` — [schema](../../../../glossary.md#term-schema) 43 and exact [specialization](../../../../glossary.md#term-specialization)/[artifact](../../../../glossary.md#term-artifact) compatibility.
- `internal.prepare` and `internal.route.portable` — existing resource and selected-plan flow.
- `internal.executable` — existing cold [binding](../../../../glossary.md#term-binding), overlap validation, worker ranges, and invocation.
- `internal.reference` — independent [conformance](../../../../glossary.md#term-conformance) realization only.

No package is added, moved, or renamed.

Type placement:

- `CpuAggregateIr` — add `Form.SUM_TO_SHAPE` and its exact structural alignment/reduction facts;
  this existing type owns aggregate generated identity.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- `CpuAggregateLowering` and nested `Geometry` — own right-aligned bound validation, selected-axis
  derivation, target-coordinate mapping, counts, scratch shape, and packed primitive geometry.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- `CpuAggregateEmitter` — owns the direct target mapping, represented-copy, modular SUM, and exact
  floating SUM bytecode; <span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>`CpuExactSumEmitter` remains the only exact-sum [limb](../../../../glossary.md#exact-superaccumulator-and-limbs)/[rounding](../../../../glossary.md#term-rounding) emitter.
- `CpuCapabilityProvider` — advertises only valid fully static resolved occurrences.
- `CpuPartitionPreparationPlan`, `CpuPartitionPreparer`, `CpuPartitionFinalizer`, and
  `CpuPreparedExecutable` — retain the existing aggregate plan/resource/binding/execution roles;
  no new type is planned.
- `CpuGeneratorSchema` and `CpuKernelSpecialization` — own schema-43 compatibility and exact entry
  shape; no new cache type is planned.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>

## Affected files

Expected production and package contracts (20):

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuExactSumEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAggregateIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAggregateLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`

Expected tests (12):

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseLedgerEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAggregateIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAggregateLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected explanatory/[planning](../../../../glossary.md#term-planning) paths (5):

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review `CpuClassFileKernelGenerator`, `CpuGeneratedKernel`, route-portable package [Javadoc](../../../../glossary.md#term-javadoc),
[Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) [SUM-to-Shape](../../../../glossary.md#sum-to-shape) contracts, shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) APIs, architecture tests,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), integration tests, and Java 26 [Gradle](../../../../glossary.md#term-gradle) configuration. Change them only if an
acceptance criterion cannot otherwise be met and the task is first updated within its ceiling.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 37 paths: the 20 production/package, 12 test, and five
documentation/[planning](../../../../glossary.md#term-planning) paths listed above. It adds no production type or package. Unused allowance
is not permission for unrelated edits.

Stop before path 38, a new production type, a new public/technically-public [API](../../../../glossary.md#term-application-programming-interface-api), another [module](../../../../glossary.md#term-module),
another [backend](../../../../glossary.md#backend), a shared contract, an architecture/build change, or any later task. Update this
specification and request direction if current code proves the in-place aggregate extension
insufficient.

## Acceptance criteria

- Exact one-[node](../../../../glossary.md#node) SUM/`SumToShapeAttrs` support for all five numeric types; every other new
  combination remains fail closed.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Fully static/resolved input, exact attrs target, and output agree; right-aligned rank and every
  bound pair are validated before route selection. Equality precedes target-one reduction.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [rank glossary](../../../../glossary.md#term-rank))</span>
- Leading, aligned-retained, aligned-preserved, scalar, equal-Shape, zero-extent, and empty-domain
  geometry matches the exact rules above with checked counts and no unresolved execution.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape); [extent glossary](../../../../glossary.md#term-extent))</span>
- Existing `CpuAggregateIr`/lowering/emitter/resource flow is extended in place; no parallel
  architecture, generic helper, or new production type appears.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- All five types preserve exact result type. Actual reductions reuse CPU 0007A1 SUM numerical and
  empty semantics; no-reduction occurrences preserve raw represented bits exactly.<span data-glossary-navigation="true"> ([represented bits glossary](../../../../glossary.md#term-represented-bits))</span>
- Dense integer-address and general long-address generated bodies cover every supported
  layout/carrier/range. Scalar and parallel-scalar results are bit-identical and one domain is
  never split.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Exactly input/output buffers are declared, output is distinct and injective, no materialization
  exists, all physical overlap is rejected before mutation, and zero-output work declares no
  workspace or generated call.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [injective glossary](../../../../glossary.md#term-injective); [materialization glossary](../../../../glossary.md#materialization))</span>
- Floating actual reductions declare exact pre-assignment per-range run-owned state; integral and
  no-reduction copy cases declare zero workspace. Scratch isolation, validation, cleanup, repeated
  use, and concurrent use are proved.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [validation glossary](../../../../glossary.md#term-validation))</span>
- Schema advances exactly 42 to 43 for the planned generated-byte and compatibility changes;
  schema-42/corrupt artifacts are safe misses and no legacy reader exists. The A1O ledger resource
  retains its historical <span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [artifacts glossary](../../../../glossary.md#term-artifact))</span>`generated-schema=42` metadata unchanged, while its integration test
  proves that schema 42 remains compatible historical evidence and separately asserts that the
  current <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [integration test glossary](../../../../glossary.md#term-integration-test))</span>`CpuGeneratorSchema.CURRENT_VERSION` is exactly 43.
- Generated descriptors, deterministic bytes, structural keys, manifests, allowed member
  references, and forbidden bridge/object/allocation/dispatch rules pass for the selected evidence
  matrix. Existing representative schema-42 semantic/code-shape controls do not regress.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [dispatch glossary](../../../../glossary.md#term-dispatch); [matrix glossary](../../../../glossary.md#term-review-matrix))</span>
- Independent semantic/[oracle](../../../../glossary.md#term-test-oracle) tests and all six five-[fork](../../../../glossary.md#term-fork-meanings) `<= 1.15x` performance cases pass.
- A separate clean documentation-focused pass finalizes every affected Javadoc/package summary,
  CPU guide, glossary impact, task evidence, master plan, and roadmap in the same overall change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Documentation explicitly reviews and records no-change conclusions where correct for Model,
  Compiler, Planning, Prepare, Runtime, public APIs, architecture/ADRs/tests, other backends,
  conformance/integration, Gradle/build, and later tasks.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [backends glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- [CPU](../../../../glossary.md#term-cpu) 0007A2 alone is `In progress` during implementation and becomes `Complete` only after every
  gate. No CPU task is [`Ready`](../../../../glossary.md#term-ready); CPU 0007A1D remains `Review needed`; completed A1 tasks remain
  unchanged; CPU 0007B and later remain `Draft` without detailed specifications.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused implementation command:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseLedgerEvidenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [CPU](../../../../glossary.md#term-cpu) suite:

```bash
./gradlew :backends:cpu:test
```

Generated-code and performance evidence:

```bash
for classfile in <fresh-evidence-root>/generated-classes/*.class; do
  javap -c -p "$classfile"
  javap -v -p "$classfile"
  sha256sum "$classfile"
done
for fork in 1 2 3 4 5; do
  java --add-modules jdk.incubator.vector -Xms1g -Xmx1g \
    -cp <fresh-evidence-root>/classes:<required-test-and-main-classpath> \
    SumToShapeGeneratedDirectBenchmark
done
```

Record the fresh evidence root, exact [compile](../../../../glossary.md#compile)/[classpath](../../../../glossary.md#term-classpath) commands, source and [artifact](../../../../glossary.md#term-artifact) checksums,
[JDK](../../../../glossary.md#term-java-development-kit-jdk)/OS/CPU [metadata](../../../../glossary.md#term-metadata), raw [fork](../../../../glossary.md#term-fork-meanings) output, per-fork ratios, [median](../../../../glossary.md#term-median)-of-fork-medians, rejected [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters),
[descriptors](../../../../glossary.md#term-descriptor), member references, and complete `javap` output. Do not reuse A1 performance results as
A2 evidence.

Documentation pass:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Also validate local Markdown links and explicit anchors, balanced fences, final newlines, trailing
whitespace, exact 37-path ceiling, package/type placement, current [schema](../../../../glossary.md#term-schema) 43, unchanged historical
`generated-schema=42` ledger metadata, the ledger test's separate historical/current schema
assertions, task/master/roadmap status, unchanged CPU 0007A1D and completed A1 records, no later
task specification, empty staging area, and preservation of unrelated concurrent work.

Repository-wide validation is deferred to CPU 0009 and [CI](../../../../glossary.md#term-continuous-integration-ci). Architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and
integration suites are not run unless implementation changes a corresponding contract; such a
need is a stop condition, not implied [scope](../../../../glossary.md#term-scope).

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0005A–0005J, especially 0005B access/[carrier](../../../../glossary.md#term-storage-carrier)/[binding](../../../../glossary.md#term-normative) contracts.
- Complete CPU 0007A, 0007A0, and 0007A1 exact aggregate semantics, direct generated bodies,
  resource lifecycle, oracle, and performance boundary.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Complete corrective CPU 0007A1A–0007A1C and 0007A1E–0007A1O accumulated schema-42 evidence.
  CPU 0007A1D remains historical <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`Review needed` and is not a [dependency](../../../../glossary.md#term-dependency) gate.
- Complete [Model](../../../../glossary.md#term-model) 0023A `SUM`/`SumToShapeAttrs` semantics and Compiler 0005B binding-constraint
  adoption.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span><span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Existing static-shape Prepare projection, staged CPU analysis/declaration/assignment/finalization,
  Runtime cold binding/run-state ownership, and Java 26 Class-File toolchain.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [toolchain glossary](../../../../glossary.md#term-toolchain))</span>

All required dependencies are complete.

## Follow-up tasks

- CPU 0007B remains the next Draft row after A2 for arg-extrema coverage. Do not create its
  detailed specification in this task.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- [CPU](../../../../glossary.md#term-cpu) 0007C–0007F and later CPU work remain ordered Draft rows.
- Dynamic/symbolic CPU execution requires a future explicit shared exact-binding contract and is
  not implied by this task.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- CPU 0009 remains the portable generated-coverage capability [checkpoint](../../../../glossary.md#term-checkpoint).

## Architecture impact

Expected impact: None.

This task implements a current [Model](../../../../glossary.md#term-model) semantic through existing [CPU](../../../../glossary.md#term-cpu)-private boundaries. If work
requires a new shared [binding](../../../../glossary.md#term-binding)/resource contract, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), public surface, [module](../../../../glossary.md#term-module)
[owner](../../../../glossary.md#term-owner), architecture rule, or another package/type architecture, stop and report the exact conflict
before editing.

## Implementation prompt

Use this prompt in a separate clean-context task/thread:

```text
You are working in /Users/phujka/IdeaProjects/Synaptik. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md,
docs/planning/backends/cpu/master-plan.md, this CPU 0007A2 task in full, its linked completed CPU
tasks, and the current Model/Compiler SUM-to-Shape plus CPU aggregate/cache/resource contracts.

Implement CPU 0007A2 exactly within its 37-path ceiling. Preserve all concurrent user work. Extend
the existing aggregate pipeline in place, advance generated compatibility exactly to schema 43,
preserve the A1O ledger resource's historical `generated-schema=42` metadata, and update its now
permitted integration test to distinguish compatible historical schema evidence from exact current
schema 43. Preserve bytecode-first optimal direct-Java parity. Do not implement dynamic binding, later
tasks, shared/architecture/build changes, another IR hierarchy, or unrelated refactors. Stop and
report any architecture, dependency, scope, type-placement, or acceptance conflict.

Run the focused command, one final CPU suite, fresh Class-File/semantic/performance evidence, and
the exact static checks. Then hand the stable diff and evidence to a separate clean-context
documentation-focused agent. That pass must follow docs/developer-guide/documentation-rules.md,
inspect final source/tests/evidence, finalize affected Javadocs/package summaries, CPU guide,
glossary impact, this task, master plan, and roadmap, and reuse successful Java evidence unless it
changes executable behavior or records a concrete reason. Do not mark Complete until all gates pass.
```

## Local decisions

- Extend the existing aggregate [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/emitter/resource path with `Form.SUM_TO_SHAPE`; a
  second family would duplicate exact SUM state and lifecycle ownership.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Equal aligned extents take precedence over target-one. This preserves the Model rule that a
  coordinate with no reduced axis is the corresponding input value, including exact represented
  bits for source-one/target-one.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Model glossary](../../../../glossary.md#term-model); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [axis glossary](../../../../glossary.md#term-axis); [represented bits glossary](../../../../glossary.md#term-represented-bits))</span>
- A no-reduction occurrence is a direct represented-bit aggregate copy with zero workspace. It
  remains one explicit distinct-output SUM artifact rather than an identity rewrite or affine
  prepared unit.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [represented-bit glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Output-cell-only partitioning preserves scalar/parallel bit identity and avoids any new
  partial/combine resource contract.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Schema 43 is required because A2 deliberately adds new generated form/mapping/copy bytes and new
  cache-visible structural/resource facts. A1O stayed at 42 precisely because it changed no
  production bytes; that rationale does not apply here.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache))</span>
- The A1O ledger resource's `generated-schema=42` fact describes the historical generated corpus,
  not the current generator version. A2 therefore permits
  `CpuPointwiseLedgerEvidenceTest.java` as its twelfth integration-test path and requires that test
  to validate historical schema-42 compatibility evidence separately from the exact current
  schema-43 assertion. The ledger resource and A1O history remain unchanged.
- No architecture, shared contract, [conformance](../../../../glossary.md#term-conformance), integration, or build change is expected.

## Known limitations

- Only fully static, already-bound [Shapes](../../../../glossary.md#shape) with resolved [layouts](../../../../glossary.md#layout) execute.
- One reduction domain remains serial; parallelism is only across complete output cells.
- The portable route remains scalar or parallel-scalar for this family; no vector or native route
  is claimed.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
- General layouts/carriers receive semantic and structural evidence, while comparative performance
  is limited to the six fixed dense cases.
- No claim extends to masked, advanced, [softmax](../../../../glossary.md#softmax--log-softmax), [normalization](../../../../glossary.md#term-normalization), or later families.

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context `01a03437-bb19-7d30-ae22-5c599a677e6a` completed executable work on
2026-08-24. The corrected focused command from this task passed after the final emitter change:

```text
./gradlew :backends:cpu:test --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuAggregateGeneratedKernelTest --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseLedgerEvidenceTest --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIrTest --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest
BUILD SUCCESSFUL in 1s; 22 actionable tasks, one executed and 21 up-to-date.
```

The single final [CPU](../../../../glossary.md#term-cpu) suite was then [run](../../../../glossary.md#term-run) exactly once after executable stabilization:

```text
./gradlew :backends:cpu:test
BUILD SUCCESSFUL in 3s; 22 actionable tasks, one executed and 21 up-to-date.
```

Development-focused aggregate, [lowering](../../../../glossary.md#lowering), [IR](../../../../glossary.md#term-intermediate-representation-ir), capability, generated-[kernel](../../../../glossary.md#kernel), preparation,
executable, [cache](../../../../glossary.md#term-cache), and ledger selections also passed. The pre-correction ledger run that required
current [schema](../../../../glossary.md#term-schema) 42 failed as recorded previously; the corrected test now independently requires
historical ledger schema 42, current generator schema 43, and `historical <= current`.

Fresh Class-File and performance evidence root:

```text
/private/tmp/synaptik-cpu-0007a2-elMsQh5P
```

Evidence was compiled and generated with:

```text
A2_CP=$(find . -type d -path '*/build/classes/java/*' | sort | paste -sd ':' -)
javac --add-modules jdk.incubator.vector -cp "$A2_CP" -d /private/tmp/synaptik-cpu-0007a2-elMsQh5P/classes backends/cpu/build/a2-evidence-src/SumToShapeEvidenceFactory.java backends/cpu/build/a2-evidence-src/SumToShapeGeneratedDirectBenchmark.java
javac -d /private/tmp/synaptik-cpu-0007a2-elMsQh5P/classes backends/cpu/build/a2-evidence-src/SumToShapeIntegralOracle.java
java --add-modules jdk.incubator.vector -cp "/private/tmp/synaptik-cpu-0007a2-elMsQh5P/classes:$A2_CP" io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.SumToShapeEvidenceFactory /private/tmp/synaptik-cpu-0007a2-elMsQh5P/generated-classes
for classfile in /private/tmp/synaptik-cpu-0007a2-elMsQh5P/generated-classes/*.class; do javap -c -p "$classfile"; javap -v -p "$classfile"; shasum -a 256 "$classfile"; done
```

The four required [artifacts](../../../../glossary.md#term-artifact) and two integral benchmark-[shape](../../../../glossary.md#shape) diagnostic artifacts are deterministic,
schema 43, and have these exact [descriptors](../../../../glossary.md#term-descriptor) and [SHA-256](../../../../glossary.md#term-sha-256) values:

| Artifact | Descriptor | SHA-256 |
| --- | --- | --- |
| [FLOAT64](../../../../glossary.md#term-float64) combined | `([D[DLjava/lang/foreign/MemorySegment;[JJJ)V` | `ebcf8cf5a554452059b9383cf4606fb28f80abf1df4fae0c0f0d6298e2ee20c9` |
| [BFLOAT16](../../../../glossary.md#term-bfloat16) general mixed | `(Ljava/lang/foreign/MemorySegment;[SLjava/lang/foreign/MemorySegment;[JJJ)V` | `4856feb45463c907b3a9201c64ca7951113403ec2a20b963bc69fd1f4a2f71c4` |
| [INT64](../../../../glossary.md#data-type--datatype) aligned | `([J[J[JJJ)V` | `3538405662a45b785f2fadd047beaa7cc29bd74baeab3708d9800cc47c81c9d4` |
| [FLOAT32](../../../../glossary.md#term-float32) represented copy | `([F[F[JJJ)V` | `d8704e5a2e91b9f6a8d0159dcb422f3239e43f5f59a1502d26349b9686b256b7` |
| [INT32](../../../../glossary.md#data-type--datatype) benchmark | `([I[I[JJJ)V` | `0e060ec1ab3fe52965dd9801455ed4af6dfdcde919f89fc505d05a3453f51d2f` |
| INT64 benchmark | `([J[J[JJJ)V` | `d7c5aa66ecdc49c48be04f1be1d0a099cdaa6169e25a9d7b62833d20da8756b5` |

Complete `javap -c/-v` output is adjacent to every class. Integral benchmark [decompilation](../../../../glossary.md#term-decompilation) was
compared directly with `javac` output for the original fast one-[accumulator](../../../../glossary.md#accumulator-type) Java [oracle](../../../../glossary.md#term-test-oracle). The
performance defect was a generated bottom-tested inner counted loop: a direct diagnostic measured
the otherwise equivalent pre-tested variants at about 154–161 microseconds and the bottom-tested
variant at about 586 microseconds. The generated integral body now uses the same pre-tested loop,
single same-width accumulator, affine `base + inner` address, one load/add per [element](../../../../glossary.md#term-element), and one
store per cell as the faster Java oracle. An interim four-accumulator direct comparator was
rejected because it slowed the clean-Java oracle; it is not accepted evidence. The integral artifacts
have no member references. Floating references are limited to typed [`MemorySegment`](../../../../glossary.md#term-memorysegment) state/access,
native-order [layout](../../../../glossary.md#layout) where required, raw-bit conversion, `Long.compareUnsigned`, and
`Long.numberOfLeadingZeros`; no generated artifact contains a Synaptik helper, bridge, allocation,
boxing, reflection, collection, `Object` entry, bootstrap, or [runtime](../../../../glossary.md#term-runtime) semantic [dispatch](../../../../glossary.md#term-dispatch).

The accepted performance command was:

```text
for fork in 1 2 3 4 5; do java --add-modules jdk.incubator.vector -Xms1g -Xmx1g -cp "/private/tmp/synaptik-cpu-0007a2-elMsQh5P/classes:$A2_CP" io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.SumToShapeGeneratedDirectBenchmark; done
```

It used the original optimal simple nested integral comparator with one accumulator. Every [fork](../../../../glossary.md#term-fork-meanings) and
[median](../../../../glossary.md#term-median)-of-fork-medians ratio passed `<= 1.15x`:

| Case | Fork 1 | Fork 2 | Fork 3 | Fork 4 | Fork 5 | Median-of-fork-medians |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FLOAT64 SUM | 0.907139379 | 0.922482808 | 0.909809078 | 0.919686346 | 0.912718408 | 0.915170586 |
| FLOAT32 SUM | 0.370789029 | 0.376636497 | 0.364385761 | 0.360382198 | 0.346378024 | 0.370789029 |
| BFLOAT16 SUM | 0.395161815 | 0.376955463 | 0.425234192 | 0.426668879 | 0.386681676 | 0.394126915 |
| INT32 SUM | 1.006048092 | 0.997510847 | 1.010224426 | 1.005217876 | 1.000896052 | 1.006048092 |
| INT64 SUM | 1.071772468 | 1.003376732 | 0.997980699 | 0.867376737 | 0.999205938 | 0.994940110 |
| FLOAT32 copy | 0.409918180 | 0.442128533 | 0.413142526 | 0.393407481 | 0.405984260 | 0.410870595 |

No [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) was rejected from the accepted run. The interrupted pre-fix five-fork command and all
four-accumulator comparator runs are rejected diagnostic evidence and do not contribute to these
ratios. Raw accepted output SHA-256 is
`ce94c9adf8c7cd0d3106b701a4aff6aceeedd562fc2af077c14aeaa5a49e16c9`; benchmark-source
SHA-256 is `d7877cccb296818ff7636a976a8eca553005f1ff1f68326ba0df563d3e725c4f`.
The environment was OpenJDK 26.0.1+8-34, macOS/Darwin 25.5.0 [ARM64](../../../../glossary.md#native-packaging-and-platform-vocabulary), Apple M3 Max, 16 cores
(12 performance and four efficiency), and 64 GB [RAM](../../../../glossary.md#term-random-access-memory-ram).

The mandatory clean documentation finalization ran in context
`01a03461-6786-75f0-ae15-798bf768ba1e`. It independently inspected the final diff, all listed
production/package contracts, tests, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) semantic [owners](../../../../glossary.md#term-owner), shared [Prepare](../../../../glossary.md#prepare)/Runtime/public
boundaries, and the complete fresh evidence set. It changed no executable Java and therefore
reused the successful focused and one final CPU-suite evidence above.

Static checks passed: CPU [Javadoc](../../../../glossary.md#term-javadoc), task-required Markdown links/anchors/fences/final-newlines and
trailing-whitespace checks, exact allowed-path/[scope](../../../../glossary.md#term-scope)/status checks, `git diff --check`,
`git diff --cached --check`, current schema exactly 43,
all fresh compatibility envelopes begin with 43, and the historical ledger remains byte-identical
at SHA-256 `41774384007bdb17b011cb8fbaae3b6928baa6ac8c04027a82540977f046031a`
with `generated-schema=42`. Staging is empty. The worktree has 22 total changed/untracked paths,
including preserved unrelated CPU 0007A1O/[planning](../../../../glossary.md#term-planning) work; A2 itself remains below its 37-path ceiling.

## Implementation notes

The existing aggregate flow now owns bound right-aligned geometry, source/[target](../../../../glossary.md#term-target) structural
[identity](../../../../glossary.md#term-identity-meanings), direct no-reduction [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) copy, general typed [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) odometers, and the
specialized dense benchmark form. Floating paths retain `CpuExactSumEmitter` as the sole exact
accumulation/conversion emitter. Integral dense loops use the optimization-relevant pre-tested
counted-loop [shape](../../../../glossary.md#shape) demonstrated by the clean Java [oracle](../../../../glossary.md#term-test-oracle). Existing preparation/finalization and
executable resource [owners](../../../../glossary.md#term-owner) required no production change.

## Completion summary

- Completed changes: Full in-place SUM-to-Shape capability validation, structural IR identity,
  checked lowering/geometry, typed general and dense generated execution, raw represented copy,
  exact resource selection, scalar reference behavior, schema-43 cache compatibility, historical
  ledger/current-schema separation, and semantic/resource/cache/lifecycle/code-shape/performance
  tests and evidence.<span data-glossary-navigation="true"> ([SUM-to-Shape glossary](../../../../glossary.md#sum-to-shape); [validation glossary](../../../../glossary.md#term-validation); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings); [lowering glossary](../../../../glossary.md#lowering); [scalar glossary](../../../../glossary.md#term-scalar); [schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [lifecycle glossary](../../../../glossary.md#lifecycle); [shape glossary](../../../../glossary.md#shape))</span>
- Implementation-owned files changed or created: `CpuCapabilityProvider.java`,
  `CpuGeneratorSchema.java`, `CpuAggregateEmitter.java`, `CpuAggregateIr.java`,
  `CpuAggregateLowering.java`, `CpuScalarReferenceKernel.java`, `CpuCapabilityProviderTest.java`,
  `CpuGeneratedKernelArtifactStoreTest.java`, `CpuKernelSpecializationTest.java`,
  `CpuAggregateGeneratedKernelTest.java`, `CpuPointwiseLedgerEvidenceTest.java`,
  `CpuPreparedExecutableTest.java`, `CpuAggregateIrTest.java`, `CpuAggregateLoweringTest.java`,
  `CpuPartitionFinalizerTest.java`, `CpuPartitionPreparerTest.java`, and this task record.
- Tests and validation: Corrected focused selection passed; the one final CPU suite passed; fresh
  semantic, Class-File, deterministic, member-reference, descriptor, compatibility, ledger,
  static, and six-case five-fork performance gates passed.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- Documentation finalization: Context `01a03461-6786-75f0-ae15-798bf768ba1e` finalized
  `CpuAggregateIr` plus the CPU provider, cache, emitter, executable, IR, lowering, and prepare
  package contracts; updated the CPU backend guide, glossary, this task, CPU master plan, and
  roadmap. The guide now records fully static right-aligned five-type support, exact floating and
  modular integral semantics, raw represented-copy behavior, output-cell-only parallelism,
  overlap/resources/lifecycle, direct general and guarded dense generated forms, schema 43, and
  the explicit dynamic/vector/native/fusion exclusions.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [vector glossary](../../../../glossary.md#term-vector); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: `CpuCapabilityProvider`, `CpuGeneratorSchema`, `CpuAggregateEmitter`,
  `CpuAggregateIr`, `CpuAggregateLowering`, and `CpuScalarReferenceKernel` accurately document the
  changed contracts. `CpuKernelSpecialization`, `CpuExactSumEmitter`, `CpuPreparedExecutable`,
  `CpuPartitionLowering`, `CpuPartitionFinalizer`, `CpuPartitionPreparationPlan`, and
  `CpuPartitionPreparer` remain accurate without edits; their behavior and ownership did not
  change. Updated package summaries are limited to the provider, cache, emitter, executable, IR,
  lowering, and prepare packages where ordinary-only or schema-42 wording became stale.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: The established Sum-to-Shape term needed only a current-boundary clarification:
  Model construction still performs no binding or execution, while CPU schema 43 executes only
  fully bound static occurrences. No new term was introduced.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [binding glossary](../../../../glossary.md#term-binding))</span>
- No-change conclusions: Model and Compiler SumToShape semantics, Planning capability contracts,
  shared Prepare analysis/finalization, Runtime and <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime))</span>[`RunState`](../../../../glossary.md#run-state--runstate), public [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs,
  `ARCHITECTURE.md`, architecture plans/ADRs/tests, other backends, backend conformance and
  integration tests, Gradle/build configuration, and later task specifications remain unchanged.
  The CPU-private implementation extends the existing aggregate path without changing those
  boundaries. CPU 0007A1D remains <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>`Review needed`; CPU 0007A1C, CPU 0007A1O, and all other
  completed A1 tasks remain `Complete`. CPU 0007B is the next ordered Draft planning follow-up;
  no CPU task is Ready and no detailed 0007B specification was created.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Unresolved implementation issues: None.
- Follow-up required: None for [CPU](../../../../glossary.md#term-cpu) 0007A2. CPU 0007B remains separate Draft [planning](../../../../glossary.md#term-planning) work.

Status: Complete
