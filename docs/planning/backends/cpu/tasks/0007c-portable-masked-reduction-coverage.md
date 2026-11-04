# Task 0007C: Portable [Masked Reduction](../../../../glossary.md#masked-reduction) Coverage

## Reader context

This task records masked `SUM`/`MEAN` through `CpuMaskedReductionIr`, `CpuMaskedReductionLowering`,
and `CpuMaskedReductionEmitter`. Their ordered boundaries are floating data, canonical `BOOL` mask,
and same-type output. Model owns directional right-aligned broadcast and selected-count meaning; CPU
proves that the mask cannot enlarge the data Shape. Generated code tests the mask before loading or
classifying data, so excluded NaNs and infinities never enter aggregation.

Here, an independent integer/rational reference is the semantic oracle; decompilation checks early
branching and direct loop structure against clean Java. `CpuExactSumEmitter` remains the
generation-time arithmetic owner. Runtime supplies isolated per-range exact-state scratch, not
persistent model state; selected count stays primitive-local, and workers own complete output cells.

Current-topic navigation (not a replacement for the historical record):

- [current directional masking and selected counts](../../../../reference/tensor/masked-index-reductions.md#masked-sum-and-mean-expressions) — Explains why the mask cannot enlarge the data Shape and why excluded values never contribute.
- [current CPU preparation and invocation handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) — Names the current analysis, shared assignment, finalization, cold binding, and Runtime participants.

## Status

Complete

## Goal

Execute exactly one fully static, resolved-[layout](../../../../glossary.md#layout), two-input masked `SUM` or [`MEAN`](../../../../glossary.md#term-mean) [occurrence](../../../../glossary.md#term-occurrence) on
the portable [CPU](../../../../glossary.md#term-cpu) route. Apply the [Model](../../../../glossary.md#term-model)'s directional right-aligned [BOOL](../../../../glossary.md#data-type--datatype)-[mask](../../../../glossary.md#term-boolean-mask) broadcast, exclude
false positions before value classification or aggregation, and produce deterministic exact
floating results through direct generated [bytecode](../../../../glossary.md#term-bytecode) with truthful exact-state resources.

## [Scope](../../../../glossary.md#term-scope)

- Recognize exactly one compiled `SUM` or [`MEAN`](../../../../glossary.md#term-mean) [node](../../../../glossary.md#node) with `MaskedReductionAttrs(axis)`, one
  floating data input, one canonical <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>[`BOOL`](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) input, and one same-floating-type output.
- Accept exactly [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16) data and output. Require a `BOOL` mask and
  the exact ordered input roles `[data, mask]`; reject integral data, `BOOL` data, and every other
  input/output type or arity.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity))</span>
- Consume the already-normalized, non-negative Model axis and validate it against the fully static
  data rank. Support exactly one axis and always remove that Dimension. Scalar data, full
  reduction, multi-axis reduction, retained Dimensions, and runtime axes are outside this task.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [rank glossary](../../../../glossary.md#term-rank); [Dimension glossary](../../../../glossary.md#dimension); [Scalar glossary](../../../../glossary.md#term-scalar); [multi-axis reduction glossary](../../../../glossary.md#multi-axis-reduction); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Require `ShapeBroadcast.broadcast(dataShape, maskShape)` to equal `dataShape` exactly. Permit a
  scalar mask, equal Shapes, missing leading mask Dimensions, and aligned mask extents equal to
  the data extent or one. Reject extra leading mask rank and any broadcast that enlarges or
  otherwise changes the data Shape. A mask such as <span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>`[batch, time]` does not broadcast to
  `[batch, time, features]`; the [Model](../../../../glossary.md#term-model)-visible `expandDims(2)` result `[batch, time, 1]` does.
- Map every data coordinate to its right-aligned mask coordinate: omitted and extent-one mask
  Dimensions use coordinate zero, while equal aligned Dimensions use the data coordinate.
  Perform this mapping directly from resolved non-negative layouts without mask materialization.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [layouts glossary](../../../../glossary.md#layout); [materialization glossary](../../../../glossary.md#materialization))</span>
- Test the canonical mask byte before loading, classifying, or accumulating its data position.
  False positions therefore cannot contribute finite values, NaNs, infinities, or signed zeros.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- For selected values, preserve the completed ordinary floating `SUM`/`MEAN` contract: exact-real
  finite summation, exact selected-count division for `MEAN`, and one round-to-nearest-ties-to-even
  conversion to the result type. Reuse the existing fixed-width exact-sum arithmetic owner rather
  than introducing a second numerical algorithm.<span data-glossary-navigation="true"> ([round-to-nearest glossary](../../../../glossary.md#floating-point-precision-and-rounding); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [owner glossary](../../../../glossary.md#term-owner))</span>
- Preserve ordinary selected special-value policy after masking: any selected NaN yields the
  CPU's canonical quiet NaN; selected opposite infinities yield that NaN; a sole infinity sign is
  preserved. The NaN payload remains unspecified by Model. Exact zero is negative only when the
  selected domain is non-empty and every selected value is negative zero; cancellation, any
  selected positive zero, or a zero selected count yields positive zero for <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [quiet NaN glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [cancellation glossary](../../../../glossary.md#term-cancellation); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`SUM`.
- Count selected true positions exactly in a checked non-negative `long` invocation local for
  each output cell. `MEAN` with selected count zero returns the same type-specific canonical quiet
  NaN used by ordinary floating aggregate generation. Otherwise divide the exact finite magnitude
  by that exact selected count before the single represented rounding. A non-empty axis with an
  all-false mask and a static zero selected-axis extent use the same zero-count rules.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding))</span>
- Lower flattened output ordinals to complete selected-axis domains. Scalar and parallel-scalar
  ranges own disjoint complete output cells; no selected domain is split or combined. Selected
  count, coordinates, flags, and arithmetic temporaries are invocation-private.
- Support arbitrary fully resolved legal non-negative data, mask, and output layouts, including
  zero-stride reads, and matching typed heap or <span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) carriers: `double[]`, `float[]`,
  or `short[]` for data/output and `byte[]` for masks, with every [heap](../../../../glossary.md#term-heap)/segment/mixed combination.
- Validate descriptor/type/Shape/layout facts, complete referenced spans, carrier type,
  accessibility, alignment, canonical mask bytes, output injectivity, output-versus-each-input
  overlap, and workspace-versus-buffer overlap before the first output write or worker submission.
  Data and mask may physically overlap because both are read-only; no contract requires a false
  read/read alias rejection.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [buffer glossary](../../../../glossary.md#term-buffer); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Declare three ordered buffer boundaries: data read, mask read, output write. Declare one
  alignment-eight `AGGREGATE_EXACT_STATE` workspace with the existing per-range exact-sum slice
  shape <span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>`8 + 8 * stateLimbCount` bytes whenever output cells exist; declare its checked total as
  one slice per simultaneously used output-cell range. Selected count is a primitive local, not
  workspace. Declare no materialization, partial, or combine resource.
- Select only scalar or parallel-scalar execution. Empty output Shapes perform no generated call,
  worker submission, read, write, or workspace use. Borrow but never close the worker group or
  caller-owned buffers.
- Add focused [CPU](../../../../glossary.md#term-cpu)-private masked-reduction [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering), and emitter owners. Preserve ordinary
  `CpuAggregateIr`/`CpuAggregateLowering`/`CpuAggregateEmitter` and arg-extrema owners. Extend the
  generation-time exact-sum emitter only as needed to accept a runtime selected-count local for
  masked <span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>`MEAN`, without changing ordinary aggregate semantics, byte [shape](../../../../glossary.md#shape), or performance.
- Emit one final, field-free, constructor-free generated class with one public static typed
  `invoke` entry. Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly once from 44 to 45; older
  envelopes are incompatible safe misses and regenerate.
- Treat an optimal clean primitive Java implementation of each specialization as the generated
  algorithm, structural-review, and performance oracle. Generated code must match its traversal,
  early mask branch, exact-state work, selected-count work, address work, and stores.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Any masked [operation](../../../../glossary.md#operation) other than current `SUM`/[`MEAN`](../../../../glossary.md#term-mean); ordinary reductions, arg extrema,
  logarithmic/statistical/norm reductions, softmax, normalization, or fusion with adjacent nodes.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [norm glossary](../../../../glossary.md#term-norm); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [fusion glossary](../../../../glossary.md#term-fusion); [nodes glossary](../../../../glossary.md#node))</span>
- Integral or BOOL data, a non-BOOL mask, numeric truthiness, noncanonical mask bytes, mask
  gradients, a differentiable mask, changed data-gradient rules, or any new public masking API.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask); [gradients glossary](../../../../glossary.md#term-gradient); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Full/multiple/runtime axes, keep-Dimension forms, scalar data, unresolved or symbolic Shapes,
  unresolved layouts, dynamic binding, or a broadcast result different from the data Shape.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [Dimension glossary](../../../../glossary.md#dimension); [scalar glossary](../../../../glossary.md#term-scalar); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [binding glossary](../../../../glossary.md#term-binding))</span>
- Mask materialization, workspace for selected counts, partial/combine states, splitting one
  selected domain, vector execution, native routes, or a generic fallback advertised as support.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [vector glossary](../../../../glossary.md#term-vector))</span>
- Generated calls to `CpuScalarReferenceKernel`, [runtime](../../../../glossary.md#term-runtime) semantic [dispatch](../../../../glossary.md#term-dispatch), generic `Object`
  carrier descriptors, allocation, boxing, reflection, method handles, <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>`invokedynamic`,
  collection/map/string dispatch, or avoidable virtual indirection in generated hot work.
- Public/shared API, Model, Compiler, Training, Runtime, Prepare contract, backend contract,
  dependency, module boundary, Gradle/toolchain, architecture, conformance, integration, tuning,
  tracing, or persistence-format changes beyond the CPU-private generator schema.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [conformance glossary](../../../../glossary.md#term-conformance); [persistence glossary](../../../../glossary.md#term-persistence); [CPU glossary](../../../../glossary.md#term-cpu); [schema glossary](../../../../glossary.md#term-schema))</span>
- Reopening CPU 0007A1D, changing completed CPU 0007A2 or CPU 0007B behavior, or creating a task
  specification for CPU 0007D or later.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007A ordinary extrema`](0007a-portable-ordinary-extrema-and-boolean-reductions.md)
- [`CPU 0007A1 numerical aggregates`](0007a1-portable-ordinary-numerical-aggregate-reductions.md)
- [`CPU 0007A1A generated self-containment`](0007a1a-generated-scalar-body-self-containment.md)
- [`CPU 0007A1C generated evidence closure`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1D native-order review`](0007a1d-native-order-segment-layout-hoisting.md)
- [`CPU 0007A1H numerical aggregate parity`](0007a1h-numerical-aggregate-residual-parity.md)
- [`CPU 0007A2 binding-aware SUM`](0007a2-portable-binding-aware-sum-to-shape-reduction.md)
- [`CPU 0007B arg extrema`](0007b-portable-arg-extrema-coverage.md)
- [`Model 0018Q masked reductions`](../../../modules/model/tasks/0018q-masked-reduction-redesign.md)
- [`Compiler 0005B reduction gradients`](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [`Prepare 0001 resource declaration`](../../../modules/prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md)
- [`Prepare 0002 finalization handoff`](../../../modules/prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Model exclusively owns masked-reduction meaning, input ordering, exact types, directional
  broadcast legality, axis normalization, result Shape/type, selected-count semantics, special
  values, empty selections, and gradient eligibility. Compiler owns capture, verification, and
  autograd construction. CPU implements forward execution only and must independently fail closed
  on any captured occurrence outside this exact subset.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [axis glossary](../../../../glossary.md#term-axis); [normalization glossary](../../../../glossary.md#term-normalization); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [CPU glossary](../../../../glossary.md#term-cpu); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- CPU analysis owns capability truth, focused lowering, access validation, execution strategy, and
  exact resource declaration before shared assignment. CPU finalization owns generated artifact
  realization. Shared Prepare remains CPU-blind; Runtime receives immutable recipes and direct
  carriers and does not interpret masking or aggregation semantics.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation); [artifact glossary](../../../../glossary.md#term-artifact); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- The existing ordinary aggregate owners remain the sole owners of one-input full/single-/multi-
  axis and SUM-to-Shape forms. Focused masked owners are required by the distinct three-boundary,
  directional-broadcast geometry. The existing exact-sum emitter remains the sole generation-time
  arithmetic owner shared by ordinary and masked floating SUM/MEAN.<span data-glossary-navigation="true"> ([SUM-to-Shape glossary](../../../../glossary.md#sum-to-shape); [owner glossary](../../../../glossary.md#term-owner); [MEAN glossary](../../../../glossary.md#term-mean))</span>
- Generated code must preserve the optimal direct clean-Java algorithm, hot-loop/dataflow shape,
  and avoidable-overhead profile. A correct helper-dispatch or generic-coordinate loop is not
  sufficient for a proved specialized path.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [dispatch glossary](../../../../glossary.md#term-dispatch); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Any required public/shared/build/architecture/conformance/integration change, new resource kind,
  mask materialization, selected-domain split/combine, or uncertainty about existing ownership is
  a stop condition. Do not edit that boundary; report and replan.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [mask glossary](../../../../glossary.md#term-boolean-mask); [materialization glossary](../../../../glossary.md#materialization); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Package impact

Existing [CPU](../../../../glossary.md#term-cpu)-private packages changed:

- `io.github.pho001.synaptik.backend.cpu` — exact capability [admission](../../../../glossary.md#term-admission).
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — focused [immutable](../../../../glossary.md#term-immutable) masked-reduction [IR](../../../../glossary.md#term-intermediate-representation-ir).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — directional broadcast validation and
  three-boundary output-cell geometry.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — focused direct typed emitter,
  exact-sum selected-count input, and generator dispatch.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — one-unit portable plan encoding.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — exact-state declaration and geometry
  propagation/finalization.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold validation and scalar/parallel
  range invocation.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent masked [oracle](../../../../glossary.md#term-test-oracle).
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema)-45 compatibility.

Packages added: None.

Type placement:

- Add exactly `CpuMaskedReductionIr`, `CpuMaskedReductionLowering`, and
  `CpuMaskedReductionEmitter` in the corresponding focused packages.
- Do not add a masked form to `CpuAggregateIr`, `CpuAggregateLowering`, or
  `CpuAggregateEmitter`; those owners encode two boundaries and uniform one-input reduction
  domains. Do not modify <span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span>`CpuArgExtremaIr`, `CpuArgExtremaLowering`, or `CpuArgExtremaEmitter`.
- Reuse `CpuExactSumEmitter` only at its generation-time arithmetic seam. Add no public facade,
  manager, registry, adapter, resource type, or runtime helper.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Affected files

Expected production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
  and directly affected package Javadoc;
- `internal/ir/CpuPortableKernelIr.java`, new `internal/ir/CpuMaskedReductionIr.java`, and
  `internal/ir/package-info.java`;
- `internal/lowering/CpuPartitionLowering.java`, new
  `internal/lowering/CpuMaskedReductionLowering.java`, and `internal/lowering/package-info.java`;
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new
  `internal/codegen/emit/CpuMaskedReductionEmitter.java`, `CpuExactSumEmitter.java`, and directly
  affected emitter package Javadoc;
- `internal/cache/CpuGeneratorSchema.java` and directly affected [cache](../../../../glossary.md#term-cache) package [Javadoc](../../../../glossary.md#term-javadoc);
- `internal/route/portable/CpuPortableRoutePlan.java` and directly affected package Javadoc;
- `internal/prepare/CpuPartitionPreparationPlan.java`, `CpuPartitionPreparer.java`,
  `CpuPartitionFinalizer.java`, and directly affected package Javadoc;
- `internal/executable/CpuPreparedExecutable.java` and directly affected package Javadoc; and
- `internal/reference/CpuScalarReferenceKernel.java` and directly affected package Javadoc.

Expected test paths:

- `CpuCapabilityProviderTest.java`, `CpuCapabilityProviderPublicShapeTest.java`, and
  `CpuInternalPackageInventoryTest.java`;
- new `CpuMaskedReductionIrTest.java`, `CpuMaskedReductionLoweringTest.java`, and
  `CpuMaskedReductionGeneratedKernelTest.java` in matching internal packages;
- `CpuClassFileKernelGeneratorTest.java`, `CpuGeneratedKernelArtifactStoreTest.java`,
  `CpuKernelSpecializationTest.java`, and `CpuPointwiseLedgerEvidenceTest.java`;
- `CpuPartitionPreparerTest.java`, `CpuPartitionFinalizerTest.java`, and
  `CpuPreparedExecutableTest.java`; and
- `CpuReferenceDifferentialTest.java`.

Expected documentation/[planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  global roadmap. Architecture, Model/API, Compiler, Training, Prepare, backend-contract,
  conformance, integration, and build documents are review-only unless a stop condition is met.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

The complete implementation, tests, and documentation may modify or create at most 45 repository
paths: at most 25 production/[Javadoc](../../../../glossary.md#term-javadoc) paths, 15 test paths, and the five named documentation and
[planning](../../../../glossary.md#term-planning) paths. Exactly three new production types are permitted: `CpuMaskedReductionIr`,
`CpuMaskedReductionLowering`, and `CpuMaskedReductionEmitter`.

A 46th path, a fourth production type, any semantic change to ordinary aggregate or arg-[extrema](../../../../glossary.md#saturation-pole-and-extremum)
[owners](../../../../glossary.md#term-owner), a second exact-sum algorithm, a shared/public/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration
path, a new resource kind, [materialization](../../../../glossary.md#materialization), or partial/combine state is a stop/replan condition.
An optional expected path may be omitted when review proves it accurate; that omission does not
authorize an unrelated replacement path.

## Acceptance criteria

- Capability supports exactly one fully static resolved-[layout](../../../../glossary.md#layout) masked `SUM`/[`MEAN`](../../../../glossary.md#term-mean) with two
  ordered inputs, exact accepted types, valid `MaskedReductionAttrs`, directional mask broadcast,
  exact axis-removed output Shape, matching gradient metadata, and injective legal output.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [injective glossary](../../../../glossary.md#term-injective))</span>
- Lowering records kind, type, normalized axis, exact data/mask/output Shapes and layouts,
  right-aligned mask-axis mapping, output count, maximum selected-domain count, exact-state limb
  count/slice bytes, and structural access facts. All checked extent, count, address, and workspace
  arithmetic fails closed on overflow.<span data-glossary-navigation="true"> ([Lowering glossary](../../../../glossary.md#lowering); [normalized axis glossary](../../../../glossary.md#normalized-axis); [limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [slice glossary](../../../../glossary.md#slice); [extent glossary](../../../../glossary.md#term-extent); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Differential tests cover both kinds and all three types; rank-one scalar output and higher
  ranks; first/middle/last axes; scalar, equal-Shape, lower-rank trailing, and explicit singleton
  mask broadcasts; dense/general/zero-stride layouts; and all heap/segment/mixed carrier patterns.<span data-glossary-navigation="true"> ([rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [scalar glossary](../../../../glossary.md#term-scalar); [rank glossary](../../../../glossary.md#term-rank); [singleton glossary](../../../../glossary.md#term-singleton); [stride glossary](../../../../glossary.md#term-stride); [heap glossary](../../../../glossary.md#term-heap))</span>
- Mask tests prove false positions are skipped before data classification using masked NaNs,
  infinities, and both signed zeros. Selected tests cover finite cancellation, one/both infinity
  signs, raw NaNs, subnormals, overflow/underflow boundaries, ties-to-even, and BFLOAT16 raw bits.<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [finite glossary](../../../../glossary.md#term-finite); [cancellation glossary](../../../../glossary.md#term-cancellation); [infinity glossary](../../../../glossary.md#term-infinity); [subnormals glossary](../../../../glossary.md#term-subnormal); [underflow glossary](../../../../glossary.md#term-underflow); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Empty selected-[axis](../../../../glossary.md#term-axis) and all-false-[mask](../../../../glossary.md#term-boolean-mask) cases produce raw [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) for `SUM` and the frozen
  type-specific canonical quiet NaN for <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [quiet NaN glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`MEAN`. Non-empty all-negative-zero selections produce
  negative zero; any selected positive zero, nonzero cancellation, or zero selection produces the
  specified zero sign. Selected counts varying by output cell are exact.<span data-glossary-navigation="true"> ([negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- The independent reference derives broadcast mapping itself and uses independent exact integer/
  rational arithmetic and explicit represented rounding. It does not call production lowering,
  packing, exact-sum emission, coordinate, or generated execution helpers.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Resource declarations contain three buffers and exactly one alignment-eight exact-state
  workspace declaration when output cells exist, sized as the existing slice shape times the used
  range count. They contain no mask copy, count workspace, partial, combine, or materialization.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [alignment glossary](../../../../glossary.md#term-alignment); [materialization glossary](../../../../glossary.md#materialization))</span>
- Cold execution validates complete spans, carrier/access/alignment, canonical mask bytes, output
  injectivity, output/input overlap, and workspace overlap before mutation or submission.
  Failures leave data, mask, output, and canaries unchanged and submit no worker work.<span data-glossary-navigation="true"> ([injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Scalar and parallel-scalar execution is raw-bit deterministic across repeated and concurrent
  runs and worker counts. Each half-open range owns complete output cells and one private scratch
  slice; no selected domain is split or shared.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval))</span>
- Generated classes are byte-deterministic, final, field-free, constructor-free, and expose one
  typed static `invoke` whose three concrete [carrier](../../../../glossary.md#term-storage-carrier) [parameters](../../../../glossary.md#term-parameter) precede [`MemorySegment`](../../../../glossary.md#term-memorysegment) scratch,
  primitive packed geometry, and `long start/end`. No `Object` or bridge [descriptor](../../../../glossary.md#term-descriptor) is permitted.
- Complete `javap -c -p` and `javap -v -p` inspection proves the mask load/branch precedes data
  load/classification, selected count increments only on true, and typed exact-state/final store
  work matches the clean-Java oracle. Generated hot work contains no Synaptik runtime/helper call,
  allocation, boxing, reflection, method handle, <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [runtime glossary](../../../../glossary.md#term-runtime); [method handle glossary](../../../../glossary.md#term-method-handle))</span>`invokedynamic`, collection/map/string lookup,
  generic semantic dispatch, or avoidable division/remainder/address recomputation.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-method-handle-lookup))</span><span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- A reviewed forbidden-member allowlist permits only necessary JDK primitive raw-bit operations
  and typed <span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`MemorySegment`/native-order layout operations. An unexpected member is a failure.
- Specialization/cache identity distinguishes kind, floating type, axis/removing form, three
  access regimes, mask rank and singleton/equal broadcast topology, carrier pattern, maximum
  domain/limb/slice shape, and every emitted-body fact. Concrete bases, offsets, stride magnitudes,
  extents that do not change emitted bytes, ranges, buffers, workers, and scratch slots stay cold.<span data-glossary-navigation="true"> ([Specialization glossary](../../../../glossary.md#term-specialization); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [topology glossary](../../../../glossary.md#term-topology))</span>
- Schema advances exactly from 44 to 45. Schema-44 and earlier artifacts safely miss; historical
  schema-42 ledger evidence remains immutable and explicitly historical.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [artifacts glossary](../../../../glossary.md#term-artifact); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Freeze six representative dense-array hot cases over [Shape](../../../../glossary.md#shape) `[128, 2048]`, axis one: [`FLOAT64`](../../../../glossary.md#term-float64),
  [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16), each for `SUM` and `MEAN`; alternate full-Shape and `[2048]`
  right-aligned masks with deterministic output-varying or shared nontrivial true patterns. Use
  non-empty complete ranges, preallocated exact state, and anti-elimination raw checksums.
- Every generated/direct pair uses identical traversal, early mask branch, carrier/layout work,
  exact-state algorithm, selected-count/division work, stores, and checksum. Each ratio passes<span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum))</span>
  `<= 1.15x` in every one of five isolated forks and for the median of fork medians. Never slow,
  pad, or route the direct oracle through production helpers.<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- The implementation passes focused tests, one final uncached CPU suite, CPU Javadoc, retained
  semantic/Class-File/determinism/checksum/five-fork evidence, exact scope, and whitespace gates.
  A distinct clean documentation context then finalizes Javadocs, CPU guide, glossary impact,
  planning status, and documentation validation without repeating stable Java/performance work.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [planning glossary](../../../../glossary.md#term-planning); [validation glossary](../../../../glossary.md#term-validation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

The implementation context runs focused tests after stabilization, then one final [CPU](../../../../glossary.md#term-cpu) suite:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuMaskedReductionGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest
./gradlew :backends:cpu:test --rerun-tasks
./gradlew :backends:cpu:javadoc
```

Retain one [immutable](../../../../glossary.md#term-immutable) evidence bundle outside the repository with exact source/probe inputs,
environment/[JVM](../../../../glossary.md#term-java-virtual-machine-jvm) facts, generated classes and [specialization](../../../../glossary.md#term-specialization) facts, complete `javap -c -p` and
`javap -v -p`, forbidden-member reports, raw semantic results, [mask](../../../../glossary.md#term-boolean-mask)-order/count/range/[canary](../../../../glossary.md#term-canary)
reports, five sequential isolated `-Xms1g -Xmx1g` [fork](../../../../glossary.md#term-fork-meanings) outputs, rejected whole [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), summaries,
inventory, and [SHA-256](../../../../glossary.md#term-sha-256) [manifest](../../../../glossary.md#term-manifest). Freeze direct oracles before timing. Each fork uses at least five
randomized warmups, nine randomized measured rounds, adaptive [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) of at least 25 ms,
deterministic inputs, randomized generated/direct order, and raw/[checksum](../../../../glossary.md#term-checksum) verification. Reject and
retain a whole fork if any case, control, checksum, or environment gate fails.

The clean documentation context reads the stabilized diff, task, source Javadocs, CPU guide,
glossary, [planning](../../../../glossary.md#term-planning) records, retained evidence, and final CPU XML. It reuses successful Java and
performance evidence unless it changes executable behavior or identifies a concrete stale-evidence
risk. It runs:

```bash
git diff --check
git diff --cached --check
git status --short -uall
```

It also validates Markdown links/anchors/fences/newlines/whitespace, exact path/type ceilings,
[schema](../../../../glossary.md#term-schema) 45 and historical schema-42 wording, CPU 0007A2/0007B Complete, CPU 0007A1D Review needed,
CPU 0007C Complete only after all gates, CPU 0007D as the next Draft task with no detailed spec,
and empty staging. Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration suites
remain deferred to CPU 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes no shared boundary.

## Dependencies

- Complete CPU 0007A1 supplies exact floating SUM/MEAN arithmetic, type-specific rounding,
  special values, signed-zero rules, exact-state layout, and complete-output-cell execution.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [MEAN glossary](../../../../glossary.md#term-mean); [rounding glossary](../../../../glossary.md#term-rounding); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [layout glossary](../../../../glossary.md#layout))</span>
- Complete CPU 0007A1A, CPU 0007A1C, and CPU 0007A1H supply generated self-containment,
  Class-File/forbidden-member discipline, direct-oracle equivalence, and five-fork performance
  gates. CPU 0007A1D remains historical <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [fork glossary](../../../../glossary.md#term-fork-meanings))</span>`Review needed` and is not a [dependency](../../../../glossary.md#term-dependency).
- Complete CPU 0007A2 and CPU 0007B supply the current preparation/finalization/cache frontier,
  schema 44, and the precedent for focused owners where boundary geometry differs.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [schema glossary](../../../../glossary.md#term-schema))</span>
- Complete Model 0018Q owns the current masked signature, types, directional right-aligned
  broadcast, false exclusion, selected-count, output, empty-selection, and gradient semantics.
  Compiler 0005B plus current normalization/autograd code owns capture, verification, and gradient
  construction. Existing backend/Prepare contracts already carry the required three buffers and
  exact workspace without change.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [signature glossary](../../../../glossary.md#term-signature); [gradient glossary](../../../../glossary.md#term-gradient); [Compiler glossary](../../../../glossary.md#term-compiler); [normalization glossary](../../../../glossary.md#term-normalization); [autograd glossary](../../../../glossary.md#autograd); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare); [carry glossary](../../../../glossary.md#term-carry); [buffers glossary](../../../../glossary.md#term-buffer))</span>

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0007D remains the next ordered Draft task and receives no detailed specification here.
- CPU 0007E and later work remain Draft in existing order. CPU 0009 or CI retains repository-wide
  capability, conformance, and integration closure.<span data-glossary-navigation="true"> ([CI glossary](../../../../glossary.md#term-continuous-integration-ci); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- CPU 0007A1D remains historical `Review needed`; resolving it is independent evidence work and
  is not a prerequisite for masked reductions.

## Architecture impact

Expected impact: None.

This task adds one [CPU](../../../../glossary.md#term-cpu)-private generated execution family behind existing [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler),
[backend](../../../../glossary.md#backend), [Prepare](../../../../glossary.md#prepare), and [Runtime](../../../../glossary.md#term-runtime) contracts. It adds no public [API](../../../../glossary.md#term-application-programming-interface-api), shared [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module),
architecture rule, resource kind, build change, [conformance](../../../../glossary.md#term-conformance) surface, or integration boundary. If
implementation proves one necessary, stop and report the uncertainty rather than editing it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007C. Work on the existing worktree
without committing, pushing, staging, resetting, reverting, deleting, or modifying unrelated
work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, CPU task 0007C and
its dependencies, Model 0018Q, Compiler 0005B and current verification/autograd code, current CPU
capability/IR/lowering/emitter/reference/route/prepare/finalization/executable/cache source and
tests, documentation rules, and the General/API-Javadoc/Planning profiles.

Implement exactly the Ready specification. Add focused CpuMaskedReductionIr,
CpuMaskedReductionLowering, and CpuMaskedReductionEmitter owners; preserve existing ordinary
aggregate and arg-extrema owners; reuse the exact-sum generation seam; advance schema 44 to 45;
and satisfy every semantic, resource, generated-Class-File, deterministic, checksum, and five-fork
optimal-direct-Java gate without slowing the direct oracle.

Stop on any task ceiling or architecture/scope conflict. Hand the stabilized diff, CPU XML, and
retained evidence to a distinct clean documentation-focused context to finalize documentation and
planning in the same change. Do not mark Complete until all gates pass. Do not commit, push, or
stage.
```

## Local decisions

- Current [Model](../../../../glossary.md#term-model) semantics narrow the older master-plan shorthand to floating-only
  [`FLOAT64`](../../../../glossary.md#term-float64)/[`FLOAT32`](../../../../glossary.md#term-float32)/[`BFLOAT16`](../../../../glossary.md#term-bfloat16) data and same-type output, [canonical](../../../../glossary.md#term-canonical-form) [`BOOL`](../../../../glossary.md#data-type--datatype) mask, one normalized
  axis that is always removed, and directional broadcast exactly to the data Shape.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [normalized axis glossary](../../../../glossary.md#normalized-axis); [Shape glossary](../../../../glossary.md#shape))</span>
- Focused masked owners preserve the two-boundary ordinary aggregate owner. The three-boundary
  mask geometry and selected-count semantics are cohesive masked responsibilities; exact floating
  accumulation/rounding remains shared through <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [rounding glossary](../../../../glossary.md#term-rounding))</span>`CpuExactSumEmitter`.
- False exclusion occurs before data load/classification. Selected values then use the ordinary
  floating SUM/MEAN special-value and signed-zero rules. Zero selected count is positive zero for
  SUM and canonical quiet NaN for MEAN.<span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [quiet NaN glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- The maximum axis extent sizes exact-state limbs. The dynamic selected count never exceeds it and
  fits the checked static domain count. Keeping that count in one primitive local is both truthful
  and optimal; it needs no resource declaration.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [extent glossary](../../../../glossary.md#term-extent))</span>
- One exact-state slice per concurrently used output-cell range is sufficient. No mask copy,
  partial, combine, or count workspace is necessary because arbitrary resolved mask addressing is
  direct and every output cell owns its complete domain.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice))</span>
- Read/read data-mask physical alias is harmless and remains permitted; output/read and
  workspace/buffer overlap remain pre-write rejections.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Schema 45 is required because a new generated family and three-boundary descriptors become
  current-compatible. Runtime binding facts stay cold where they do not change bytes.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [descriptors glossary](../../../../glossary.md#term-descriptor); [Runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding))</span>

## Known limitations

- Only fully static resolved-layout, single-axis, axis-removing masked SUM/MEAN is supported.
  Dynamic/symbolic execution, keep-Dimension/full/multi-axis forms, vector execution, and native
  routes remain unsupported.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [axis glossary](../../../../glossary.md#term-axis); [MEAN glossary](../../../../glossary.md#term-mean); [Dimension glossary](../../../../glossary.md#dimension); [vector glossary](../../../../glossary.md#term-vector))</span>
- Parallelism is only across independent output cells. Small results may execute on the invoking
  thread even when a worker group exists.
- Canonical NaN bits are a CPU-private deterministic realization of Model's unspecified payload,
  not a new portable payload guarantee.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- The five-fork gate applies to the frozen cases, environment, and protocol; it is not a universal
  speed promise or Runtime tuning input.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context `01a03758-2071-7d73-b91d-a680d2783cfb` completed the stabilized executable
work and retained its [immutable](../../../../glossary.md#term-immutable) evidence under `/tmp/synaptik-cpu-0007c-evidence`. Documentation
context `/root` independently read the architecture and [planning](../../../../glossary.md#term-planning) contracts, documentation rules
and General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/[Backend](../../../../glossary.md#backend)-guide/Planning/Example profiles, task dependencies and [Model](../../../../glossary.md#term-model) 0018Q,
the complete affected diff and tests, affected Javadocs and all nine candidate package summaries,
the [CPU](../../../../glossary.md#term-cpu) guide, glossary, CPU master plan, roadmap, and retained evidence.

Implementation evidence reused without rerunning Java or performance suites:

- the exact focused eight-class command passed after stabilization;
- final uncached `./gradlew :backends:cpu:test --rerun-tasks` passed 398 tests with one expected
  skip and zero failures or errors; retained XML and the final HTML index confirm the result;
- all six [FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16) SUM/[MEAN](../../../../glossary.md#term-mean) cases passed five isolated `-Xms1g -Xmx1g` forks and
  their median-of-five gates at <span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median))</span>`<= 1.15x`; the worst individual ratio was `0.980505473x` and the
  worst median was `0.968155614x`;
- complete retained `javap -c -p` and `javap -v -p` output covers all six deterministic generated
  classes. `mask-order-report.csv` records mask load and false branch before the later data load,
  and <span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>`forbidden-members.txt` records zero forbidden hits;
- every entry in `SHA256SUMS` verified, and the [manifest](../../../../glossary.md#term-manifest) file's [SHA-256](../../../../glossary.md#term-sha-256) is
  `40be4f93a3e81cef38407f69ae1a008e38ffc05b5fe572bcb59aa933ba158526`; and
- no executable Java changed after that retained evidence. The documentation pass therefore did
  not repeat stable Java or performance suites.

Documentation and final repository validation:

- `./gradlew :backends:cpu:javadoc` passed after the Javadoc/package edits with exactly the two
  expected `jdk.incubator.vector` notices and no doclint warning. A final rerun after a grammar-only
  package-summary correction passed with the same two notices and no doclint warning. Generated pages for
  `CpuMaskedReductionIr`, `CpuMaskedReductionLowering.Layout`/`Geometry`,
  `CpuMaskedReductionEmitter`, and all nine affected package summaries were inspected for record
  parameters, accessor contracts, directional broadcast, early false exclusion, selected count,
  exact-state ownership, schema 45, and unsupported-internal boundaries;<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [schema glossary](../../../../glossary.md#term-schema))</span>
- the final local Markdown validator resolved 848 local file links, including 303 heading-anchor
  links, across the five changed documentation/planning files. Balanced-fence, final-newline,
  CRLF, trailing-whitespace, and glossary-anchor checks passed;
- scope validation found exactly 42 changed or created repository paths: 25 production/Javadoc,
  12 test, and the five named documentation/planning paths. Exactly three new production types
  exist: <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`CpuMaskedReductionIr`, `CpuMaskedReductionLowering`, and
  `CpuMaskedReductionEmitter`; no fourth type or later task specification exists;
- schema/status/order checks confirm schema 45 is current, schema-42 retained material is labeled
  historical, CPU 0007A2/0007B/0007C are Complete, CPU 0007A1D remains Review needed, and CPU
  0007D is the next ordered Draft task with no detailed specification; and
- `git diff --check`, `git diff --cached --check`, and final `git status --short -uall` passed the
  required hygiene review. Staging is empty; no commit, push, stage, reset, revert, or deletion
  occurred.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>

Architecture and adjacent-contract review found no contradiction. `ARCHITECTURE.md`, the current
architecture plan, ADRs, and architecture tests remain accurate because the change stays inside
the existing concrete-backend [lowering](../../../../glossary.md#lowering)/finalization/execution [owners](../../../../glossary.md#term-owner) and changes no [dependency](../../../../glossary.md#term-dependency) or
[module](../../../../glossary.md#term-module) boundary. Model/[Tensor](../../../../glossary.md#tensor) and API contracts remain accurate because Model 0018Q already owns
the exact [signature](../../../../glossary.md#term-signature), directional broadcast, exclusion, selected-count, and zero-selection meaning.
[Compiler](../../../../glossary.md#term-compiler) and Training remain accurate because no [capture](../../../../glossary.md#term-capture-meanings), [gradient](../../../../glossary.md#term-gradient), [optimizer](../../../../glossary.md#term-optimizer), or orchestration
behavior changed. Shared [Prepare](../../../../glossary.md#prepare)/backend-contract remain accurate because the existing three
[buffer](../../../../glossary.md#term-buffer) declarations and `AGGREGATE_EXACT_STATE` resource [carry](../../../../glossary.md#term-carry) the exact requirements without a
new contract. Backend [conformance](../../../../glossary.md#term-conformance)/integration remain deferred to CPU 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci); [Gradle](../../../../glossary.md#term-gradle)/build and
other modules remain unchanged because no shared surface, dependency, or [toolchain](../../../../glossary.md#term-toolchain) fact changed.

## Implementation notes

- Added the focused immutable masked-reduction IR, lowering geometry, and direct typed Class-File
  emitter while preserving the ordinary aggregate and arg-extrema owners.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- Extended exact-sum generation only at the runtime-selected-count finalization seam. Ordinary
  aggregate semantics, generated bytes, and performance evidence remain unchanged.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- Admitted only fully static resolved-layout FLOAT64/FLOAT32/BFLOAT16 SUM/MEAN with one normalized
  removed axis and a canonical BOOL mask whose directional right-aligned broadcast equals the data
  Shape. False positions branch around all data load and classification work.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [MEAN glossary](../../../../glossary.md#term-mean); [axis glossary](../../../../glossary.md#term-axis); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask); [Shape glossary](../../../../glossary.md#shape))</span>
- Declared ordered data/mask/output buffers and one per-range exact-state workspace when output
  cells exist. Selected count remains invocation-local; no mask materialization, count workspace,
  partial state, or combine state was added.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [materialization glossary](../../../../glossary.md#materialization))</span>
- Finalized the three affected new-type Javadocs and all nine candidate CPU package summaries. All
  nine summaries required masked-family or current-schema wording; none was left unchanged. The
  other thirteen changed production classes' existing Javadocs were independently reviewed and
  remain accurate because their draft updates already cover the added optional geometry, resource,
  validation, finalization, or dispatch facts without changing their established ownership,
  lifecycle, nullability, threading, side effects, or failure contracts.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [schema glossary](../../../../glossary.md#term-schema); [validation glossary](../../../../glossary.md#term-validation); [dispatch glossary](../../../../glossary.md#term-dispatch); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Updated the CPU backend guide and existing masked-reduction glossary entry. No duplicate glossary
  term was added.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

## Completion summary

- Completed changes: Implemented and documented the bounded portable masked SUM/MEAN family with
  directional mask broadcast, early exclusion, exact selected counts, exact floating state,
  direct generated bytecode, and schema 45 compatibility.<span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean); [mask glossary](../../../../glossary.md#term-boolean-mask); [bytecode glossary](../../../../glossary.md#term-bytecode); [schema glossary](../../../../glossary.md#term-schema))</span>
- Files changed or created: 25 production/Javadoc paths, 12 test paths, and the five authorized
  documentation/planning paths; 42 repository paths total, with exactly three new production
  types.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: Reused the stabilized focused-suite, 398-test uncached CPU, six-class
  Class-File, checksum, and five-fork performance evidence; final CPU Javadoc, generated-page,
  Markdown link/anchor/format, scope/type, schema/status/order/history, and Git hygiene checks
  passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu); [checksum glossary](../../../../glossary.md#term-checksum); [fork glossary](../../../../glossary.md#term-fork-meanings); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Clean documentation context `/root` completed the independent
  finalization in the same overall change without changing executable behavior or repeating stable
  Java/performance suites.
- Documentation impact: CPU package summaries, backend guide, glossary, task, CPU master plan, and
  roadmap now describe the exact current boundary and retain explicit planned limitations.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Javadoc review: All changed Java contracts were reviewed; the six known doclint warnings were
  fixed and only the two expected incubator-module notices remain.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Glossary impact: The existing masked-reduction entry now distinguishes Model semantics from the
  current bounded CPU realization; no new term was necessary.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for CPU 0007C. CPU 0007D remains the next ordered Draft task and must be
  specified separately before implementation.

Status: Complete
