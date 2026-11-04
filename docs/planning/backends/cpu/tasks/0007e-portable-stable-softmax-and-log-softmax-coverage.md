# Task 0007E: Portable Stable [Softmax](../../../../glossary.md#softmax--log-softmax) and [Log-Softmax](../../../../glossary.md#softmax--log-softmax) Coverage

## Reader context

This task records stable forward execution of explicit SOFTMAX and LOG_SOFTMAX nodes, not
recognition of equivalent arithmetic graphs. Model owns kind, axis, Shape, type, and gradient
metadata; those descriptors are not storage carriers. Compiler preserves the occurrence.
`CpuSoftmaxIr` and `CpuSoftmaxLowering` describe complete normalization slices, while
`CpuSoftmaxEmitter` generates a kind-specialized body over bound arrays or segments.

Here, `CpuSoftmaxInputValidator` checks the CPU-private finite domain before writes or worker
submission. Prepare assigns declared buffers; Runtime invokes the prepared recipe without semantic
dispatch. The clean Java comparison is the structural oracle for algorithm and overhead; independent
mathematical checks are the semantic oracle. Neither the zero-workspace execution nor its primitive
locals introduce persistent model state.

Current-topic navigation (not a replacement for the historical record):

- [current first-class softmax expression meaning](../../../../reference/tensor/normalization.md#softmax-expressions) — Separates metadata construction and ideal slice meaning from backend-specific admission and execution.
- [current CPU preparation and invocation handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) — Names the current analysis, shared assignment, finalization, cold binding, and Runtime participants.

## Status

Complete

## Goal

Execute exactly one fully static, resolved-[layout](../../../../glossary.md#layout), first-class [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax) or `LOG_SOFTMAX`
[occurrence](../../../../glossary.md#term-occurrence) on the portable [CPU](../../../../glossary.md#term-cpu) route. Preserve its one-[axis](../../../../glossary.md#term-axis), [shape](../../../../glossary.md#shape)-preserving [Model](../../../../glossary.md#term-model) [identity](../../../../glossary.md#term-identity-meanings) and
evaluate admitted [finite](../../../../glossary.md#term-finite) non-empty [normalization](../../../../glossary.md#term-normalization) slices with direct stable max-shift generated
loops that match optimal clean Java implementations.

## [Mental model](../../../../glossary.md#term-mental-model)

```text
one first-class softmax node
  -> CPU validates one static shape-preserving slice geometry
  -> CPU rejects the unsettled numerical-edge domain before any write
  -> one kind-specialized three-pass generated body
       maximum -> shifted exponential sum -> normalized stores
  -> scalar or complete-slice parallel-scalar execution
```

The [operation](../../../../glossary.md#operation) remains [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax) or `LOG_SOFTMAX` from [Model](../../../../glossary.md#term-model) [capture](../../../../glossary.md#term-capture-meanings) through [CPU](../../../../glossary.md#term-cpu) [lowering](../../../../glossary.md#lowering). CPU
never recognizes a subtraction/exponential/reduction/division/[log](../../../../glossary.md#term-log) [graph](../../../../glossary.md#graph) as this stable family.

## [Scope](../../../../glossary.md#term-scope)

- Recognize exactly one compiled one-input/one-output [node](../../../../glossary.md#node) whose kind is `SoftmaxKind.SOFTMAX` or
  `SoftmaxKind.LOG_SOFTMAX` and whose attributes are exactly `SoftmaxAttrs`.
- Preserve first-class semantic identity. Admit only the explicit kind and never infer, fuse, or
  reconstruct stable softmax from decomposed pointwise or reduction graphs.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [pointwise glossary](../../../../glossary.md#term-pointwise); [graphs glossary](../../../../glossary.md#graph))</span>
- Accept exactly [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16), with identical input/output type, exact
  Shape, and gradient-eligibility metadata. Reject integral, BOOL, cross-type, wrong-attribute,
  wrong-arity, mismatched-Shape, or mismatched-eligibility occurrences.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [BOOL glossary](../../../../glossary.md#data-type--datatype); [arity glossary](../../../../glossary.md#term-arity))</span>
- Require a fully static rank-positive Shape, a normalized in-range axis, fully resolved
  non-negative input/output layouts, checked representable spans, and an injective writable
  output. Support arbitrary legal offsets and positive or zero input strides.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [layouts glossary](../../../../glossary.md#layout); [injective glossary](../../../../glossary.md#term-injective); [strides glossary](../../../../glossary.md#term-stride))</span>
- Require the selected axis extent to be strictly positive. A zero selected extent is statically
  ineligible because Model and Compiler deliberately do not define its forward numerical result.
  If another axis has zero extent, the exact shape-preserving output is empty and execution is a
  no-op with no read, write, validation scan, generated call, or worker submission.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [validation glossary](../../../../glossary.md#term-validation); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Define a CPU-private admitted runtime value subset: every represented input must be finite, and
  every <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [runtime glossary](../../../../glossary.md#term-runtime); [finite glossary](../../../../glossary.md#term-finite))</span>`value - sliceMaximum` used by the selected algorithm must be finite. Validate the complete
  input and all such shifts before the first output write or worker submission. Reject NaN, either
  infinity, or a finite extreme pair whose subtraction overflows. This is fail-closed CPU
  execution policy, not Model semantics or a cross-backend numerical promise.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [backend glossary](../../../../glossary.md#backend))</span>
- Traverse normalization slices in canonical logical row-major order over all non-selected axes,
  and traverse each selected axis in increasing logical coordinate order. Retain the first value
  on an equal-maximum tie. Flattened generated <span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`start`/`end` bounds count complete slices, never
  elements within a slice.<span data-glossary-navigation="true"> ([elements glossary](../../../../glossary.md#term-element))</span>
- Use three direct passes per admitted [slice](../../../../glossary.md#slice). Pass one finds the maximum. Pass two accumulates
  `exp(value - maximum)` with the type-specific primitive compensated state defined below. Pass
  three stores:
  - [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax): `exp(value - maximum) / sum`;
  - `LOG_SOFTMAX`: `(value - maximum) - log(sum)`.
- Use binary64 maxima, shifts, transcendental results, and compensated sums for FLOAT64 and
  FLOAT32. For BFLOAT16, decode to binary32, use binary32 maxima/shifts and compensated binary32
  accumulation, narrow each <span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`Math.exp`/`Math.log` result to binary32 at the selected formula point,
  and encode the final result once to BFLOAT16 with round-to-nearest-ties-to-even. This follows the
  CPU plan's current FLOAT32 default for numerically sensitive 16-bit work while selecting a wider
  CPU-private policy for FLOAT32; neither choice is public promotion or cross-backend bitwise
  behavior. Inline BFLOAT16 conversion through existing carrier-emission patterns.<span data-glossary-navigation="true"> ([round-to-nearest glossary](../../../../glossary.md#floating-point-precision-and-rounding); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Use primitive locals only for maxima, compensated sums, shifted values, logarithms, coordinates,
  counts, and addresses. Add no workspace, materialization, partial result, combine state, saved
  value, or resource kind.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Support matching typed [heap](../../../../glossary.md#term-heap) arrays and native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment) carriers—`double[]`,
  `float[]`, or `short[]`—for every heap/segment/mixed input/output pair. Perform one cold carrier
  dispatch/binding only; generated hot work uses the exact typed entry and has no storage discovery
  or carrier switch.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [binding glossary](../../../../glossary.md#term-binding))</span>
- Reject every input/output overlap. Validate carrier form, liveness, thread access, native order,
  alignment, complete spans, output writability/injectivity, and overlap before value validation;
  value validation then completes before mutation or worker submission.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [alignment glossary](../../../../glossary.md#term-alignment); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Add one focused softmax IR, one focused geometry/lowering owner, one cohesive emitter for both
  generation-time kinds, one focused pre-execution finite-domain validator, and one independent
  reference owner. Reuse existing CPU-private geometry, carrier, loop, preparation, binding, and
  artifact patterns without adding either kind to aggregate IR/emission or growing a god emitter.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [owner glossary](../../../../glossary.md#term-owner); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Emit one final, field-free, constructor-free generated class with one public static typed
  `invoke` entry. Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly once from 46 to 47 because
  the new IR family, emitted bodies, member set, and compatibility facts change generated artifact
  meaning. Schema 46 and earlier envelopes are incompatible safe misses; retained schema-42
  evidence remains historical.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema))</span>
- Use well-written optimal clean Java loops with the same validation work, pass count, traversal,
  arithmetic, Math calls, carrier/layout addressing, and narrowing as the generated/direct oracle.
  Generated code may call only required JDK primitive, <span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [oracle glossary](../../../../glossary.md#term-test-oracle); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`Math`/`StrictMath`, and typed
  `MemorySegment` operations and must call no Synaptik numerical or reference helper.

## Out of [scope](../../../../glossary.md#term-scope)

- Decomposing either first-class kind; recognizing a decomposed graph; fusion with adjacent
  operations; masked, scaled, temperature-controlled, biased, causal, sparse, sampled, or loss
  softmax variants.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [fusion glossary](../../../../glossary.md#term-fusion); [loss glossary](../../../../glossary.md#loss); [softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- Defining portable semantics for NaN, infinity, finite shift overflow, or a zero selected-axis
  extent. Those inputs fail closed until a Model/config/conformance owner settles them.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [finite glossary](../../../../glossary.md#term-finite); [overflow glossary](../../../../glossary.md#term-overflow); [axis glossary](../../../../glossary.md#term-axis); [extent glossary](../../../../glossary.md#term-extent); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [Model glossary](../../../../glossary.md#term-model); [conformance glossary](../../../../glossary.md#term-conformance); [owner glossary](../../../../glossary.md#term-owner))</span>
- Dynamic or symbolic Shapes, unresolved layouts, scalar inputs, runtime axes, negative stored
  axes, selected-axis counts that overflow current checked geometry, or output aliasing.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [runtime glossary](../../../../glossary.md#term-runtime); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- Integral/BOOL execution, FLOAT16, cross-type input/output, public accumulator selection,
  approximation modes, relaxed math, native/vendor routes, or Vector API kernels.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16); [accumulator glossary](../../../../glossary.md#accumulator-type); [Vector API glossary](../../../../glossary.md#term-vector-api); [kernels glossary](../../../../glossary.md#kernel))</span>
- Splitting one slice across ranges, partial reductions, combine trees, workspace, materialization,
  saved forward values, backward execution, gradient-policy changes, or Runtime semantic dispatch.<span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [materialization glossary](../../../../glossary.md#materialization); [gradient glossary](../../../../glossary.md#term-gradient); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Generated calls to CPU validation/reference implementations or Synaptik arithmetic helpers;
  allocation, boxing, reflection, method handles, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`invokedynamic`, generic `Object` descriptors,
  collection/map/string lookup, or avoidable virtual dispatch in generated hot work.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-method-handle-lookup))</span><span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- A universal bitwise transcendental promise. Exact generated/direct algorithm identity is
  required for admitted values; mathematical accuracy uses the frozen tolerance matrix.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [tolerance glossary](../../../../glossary.md#term-tolerance); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Public/shared API, Model, Compiler, Training, Runtime, Prepare contract, backend-contract,
  dependency, module boundary, Gradle/toolchain, architecture, conformance, integration, tuning,
  tracing, or persistence-format changes beyond the CPU-private generator schema.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [persistence glossary](../../../../glossary.md#term-persistence); [schema glossary](../../../../glossary.md#term-schema))</span>
- Reopening [CPU](../../../../glossary.md#term-cpu) 0007A1D or creating a detailed CPU 0007F-or-later specification.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007D advanced reductions`](0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md)
- [`Model 0016I softmax semantics`](../../../modules/model/tasks/0016i-softmax-semantic-kinds-and-attributes.md)
- [`Model 0016J softmax expressions`](../../../modules/model/tasks/0016j-softmax-tensor-expressions.md)
- [`Compiler 0005B softmax inference and gradients`](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [`Prepare 0001 resource declaration`](../../../modules/prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md)
- [`Prepare 0002 finalization handoff`](../../../modules/prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Model owns the two first-class kinds, normalized axis, ideal mathematical relationship, Shape,
  type, and gradient eligibility. Compiler owns capture, revalidation, and formulas using the
  exact forward output. CPU owns only truthful forward capability, its finite admitted subset,
  lowering, algorithm, access validation, resource truth, and generated realization.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [normalized axis glossary](../../../../glossary.md#normalized-axis); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [finite glossary](../../../../glossary.md#term-finite); [lowering glossary](../../../../glossary.md#lowering); [validation glossary](../../../../glossary.md#term-validation))</span>
- [CPU](../../../../glossary.md#term-cpu) analysis must select and lower only an explicit `SoftmaxKind` occurrence. Neither Compiler
  nor CPU may infer stable softmax from an equivalent-looking decomposed graph.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [graph glossary](../../../../glossary.md#graph))</span>
- Model 0016I/0016J explicitly defer forward finite-precision and numerical-edge policy; Compiler
  0005B adds no such policy. CPU therefore fails closed outside finite, positive-width slices and
  does not claim that rejection or any finite rounding choice is portable semantics.<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- CPU analysis and preparation own all static eligibility and exact resources before shared slot
  assignment. CPU executable binding owns representation validation; its focused validator owns
  the complete value-domain check before mutation/submission. Shared Prepare remains CPU-blind and
  Runtime invokes the cold-bound prepared recipe without operation dispatch.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Generated code must preserve the optimal clean Java algorithm, pass structure, dataflow, and
  avoidable-overhead profile. A generic bridge, reference call, per-element callback, packed kind
  dispatch, or decomposition is not an implementation of this specialized family.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [element glossary](../../../../glossary.md#term-element); [callback glossary](../../../../glossary.md#term-callback); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- Any needed public/shared/build/architecture/conformance/integration change, new resource kind,
  materialization, partial/combine protocol, or unresolved numerical ownership is a stop condition.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [materialization glossary](../../../../glossary.md#materialization); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Package impact

Existing [CPU](../../../../glossary.md#term-cpu)-private packages changed:

- `io.github.pho001.synaptik.backend.cpu` — exact capability [admission](../../../../glossary.md#term-admission).
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — [immutable](../../../../glossary.md#term-immutable) [softmax](../../../../glossary.md#softmax--log-softmax) [identity](../../../../glossary.md#term-identity-meanings).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — static [slice](../../../../glossary.md#slice) geometry.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct kind-specialized bodies.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — complete pre-write finite-domain
  validation and existing bound invocation orchestration.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [validation glossary](../../../../glossary.md#term-validation); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent differential [oracle](../../../../glossary.md#term-test-oracle).
- existing `route.portable`, [`prepare`](../../../../glossary.md#prepare), and [`cache`](../../../../glossary.md#term-cache) packages — realization, declaration,
  finalization, and schema-47 compatibility.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>

Packages added: None.

Type placement:

- Add `CpuSoftmaxIr` in `internal.ir` for kind, type, normalized axis, algorithm version, access,
  pass, and structural specialization identity.<span data-glossary-navigation="true"> ([normalized axis glossary](../../../../glossary.md#normalized-axis); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Add `CpuSoftmaxLowering` in `internal.lowering` for exact Shape/layout, slice count/width,
  logical traversal, access geometry, and zero-resource derivation.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- Add `CpuSoftmaxEmitter` in `internal.codegen.emit`; generation-time kind selection produces one
  direct three-pass body without runtime semantic dispatch.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Add `CpuSoftmaxInputValidator` in `internal.executable`; it owns only direct typed complete-input
  finite/shift validation before execution and does not calculate outputs.
- Add `CpuSoftmaxReferenceKernel` in `internal.reference`; it independently derives coordinates,
  validates the admitted domain, and calculates expected results without production lowering,
  emission, binding, or generated helpers.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding))</span>
- Add no [facade](../../../../glossary.md#term-facade), manager, [registry](../../../../glossary.md#term-registry), generic [normalization](../../../../glossary.md#term-normalization) interpreter, or aggregate-[owner](../../../../glossary.md#term-owner) change.

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) paths:

- `CpuCapabilityProvider.java` and [CPU](../../../../glossary.md#term-cpu) package Javadoc;
- `internal/ir/CpuPortableKernelIr.java`, new `CpuSoftmaxIr.java`, and [IR](../../../../glossary.md#term-intermediate-representation-ir) package Javadoc;
- `internal/lowering/CpuPartitionLowering.java`, new `CpuSoftmaxLowering.java`, and lowering package
  Javadoc;<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new `CpuSoftmaxEmitter.java`, existing
  carrier/loop emission seams only where required, and emitter package Javadoc;<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `internal/cache/CpuGeneratorSchema.java`, `CpuKernelSpecialization.java` only if exact structural
  identity requires it, and cache package Javadoc;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [cache glossary](../../../../glossary.md#term-cache))</span>
- `internal/route/portable/CpuPortableRoutePlan.java` and package Javadoc;
- `internal/prepare/CpuPartitionPreparationPlan.java`, `CpuPartitionPreparer.java`,
  `CpuPartitionFinalizer.java`, and package Javadoc;
- `internal/executable/CpuPreparedExecutable.java`, new `CpuSoftmaxInputValidator.java`, and
  executable package Javadoc; and
- new `internal/reference/CpuSoftmaxReferenceKernel.java`, the minimal existing differential seam
  if required, and reference package Javadoc.

Expected test paths:

- capability and internal-package inventory tests;
- new `CpuSoftmaxIrTest`, `CpuSoftmaxLoweringTest`, `CpuSoftmaxGeneratedKernelTest`,
  `CpuSoftmaxInputValidatorTest`, and `CpuSoftmaxReferenceTest`;
- generated-class shape, specialization/cache, portable route, prepare/finalization, executable,
  and reference-differential tests only where their exact inventories or behavior change; and<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [specialization glossary](../../../../glossary.md#term-specialization); [prepare glossary](../../../../glossary.md#prepare))</span>
- the historical pointwise-ledger test only if a current-schema assertion requires it; its
  historical evidence resource must not change.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [schema glossary](../../../../glossary.md#term-schema))</span>

Expected documentation/[planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  global roadmap. Architecture, Model/Tensor API, Compile API, Training API, shared Prepare,
  backend-contract, conformance, integration, and build files are review-only unless a stop
  condition is reached.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compile glossary](../../../../glossary.md#compile); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

The complete implementation, tests, and documentation may modify or create at most 49 repository
paths: at most 29 production/[Javadoc](../../../../glossary.md#term-javadoc) paths, 15 test paths, and the five named documentation and
[planning](../../../../glossary.md#term-planning) paths. Exactly five new production types are permitted: the [IR](../../../../glossary.md#term-intermediate-representation-ir), lowerer, emitter,
validator, and independent reference [owner](../../../../glossary.md#term-owner) listed above.

A 50th path, a sixth production type, a second numerical emitter, an aggregate-emitter or
advanced-reduction semantic change, a public/shared/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration
path, [materialization](../../../../glossary.md#materialization), workspace/resource-kind addition, or partial/combine state is a stop/replan
condition. An omitted expected path does not authorize an unrelated one.

## Acceptance criteria

- Capability admits exactly both explicit [softmax](../../../../glossary.md#softmax--log-softmax) kinds with `SoftmaxAttrs`, one input/output,
  FLOAT64/FLOAT32/BFLOAT16 identity, exact static non-scalar Shape/eligibility, normalized axis,
  positive selected extent, resolved non-negative layouts, and injective output. Every excluded
  occurrence fails closed.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [identity glossary](../../../../glossary.md#term-identity-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape); [normalized axis glossary](../../../../glossary.md#normalized-axis); [extent glossary](../../../../glossary.md#term-extent); [layouts glossary](../../../../glossary.md#layout); [injective glossary](../../../../glossary.md#term-injective); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- A graph composed from maximum/subtraction/exponential/sum/division/log nodes remains those
  ordinary nodes and is never admitted or lowered as <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [log glossary](../../../../glossary.md#term-log); [nodes glossary](../../../../glossary.md#node))</span>`CpuSoftmaxIr`; source-backed tests lock this.
- Lowering records exact kind/type/axis, Shape, slice count, positive slice width, logical axis
  traversal, input/output access facts, algorithm/pass identity, carrier pattern, and zero
  workspace. Checked count/address arithmetic precedes declaration.<span data-glossary-navigation="true"> ([Lowering glossary](../../../../glossary.md#lowering); [slice glossary](../../../../glossary.md#slice); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Differential coverage includes both kinds and all three types; first/middle/last axes; rank one
  and higher; selected extent one and larger; other-axis zero extents; dense/general/zero-stride
  input layouts; non-contiguous injective outputs; offsets; and all four heap/segment/mixed pairs.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [stride glossary](../../../../glossary.md#term-stride); [contiguous glossary](../../../../glossary.md#term-contiguous); [heap glossary](../../../../glossary.md#term-heap))</span>
- CPU-private rejection tests cover raw NaNs, both infinities, mixed signs, and finite extreme
  pairs whose subtraction is infinite. The complete input and shifts are checked before any write
  or worker submission; output/canaries remain unchanged on failure.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [finite glossary](../../../../glossary.md#term-finite))</span>
- Admitted finite tests cover signed zeros, subnormals, equal maxima, large common offsets, extreme
  but finite shifts, underflowed probabilities, singleton slices, and strongly skewed slices.
  These lock CPU algorithm behavior without describing it as cross-backend semantics.<span data-glossary-navigation="true"> ([subnormals glossary](../../../../glossary.md#term-subnormal); [singleton glossary](../../../../glossary.md#term-singleton); [backend glossary](../../../../glossary.md#backend))</span>
- Mathematical accuracy uses an independent high-precision or StrictMath-based oracle and a frozen
  table: FLOAT64 outputs are within 8 result ulps, FLOAT32 within 4 result ulps, and BFLOAT16 within
  one represented BFLOAT16 ulp. Generated/direct results for the identical algorithm are raw-bit
  equal; rejected values and zero-width slices are never tolerance-tested as outputs.<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [oracle glossary](../../../../glossary.md#term-test-oracle); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- Resource declarations contain exactly one input and one output, with zero workspace,
  materialization, partial, combine, or hidden scratch. Empty outputs declare their existing
  boundary spans but perform no value scan or generated/worker call.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Cold binding validates carrier/access/alignment/spans, output injectivity, and complete overlap
  rejection before the focused value validator. Every failure precedes mutation/submission.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [alignment glossary](../../../../glossary.md#term-alignment); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Scalar and parallel-scalar outputs are raw-bit deterministic across legal complete-slice ranges,
  worker counts, repeated runs, and concurrent runs. No range divides a normalization slice.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- Generated classes are deterministic, final, field-free, constructor-free, and expose one typed
  static `invoke` with concrete input/output carriers, packed primitive geometry, and `long
  start/end`; no `Object`, bridge [descriptor](../../../../glossary.md#term-descriptor), kind [argument](../../../../glossary.md#term-argument), or validator/reference call exists.
- Complete `javap -c -p` and `javap -v -p` inspection proves the three direct passes, inlined
  carrier/address work, one store per output position, generation-time kind specialization, and
  absence of allocation, boxing, reflection, method handles, <span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>`invokedynamic`, collections/maps/
  strings, semantic dispatch, and avoidable coordinate/address recomputation.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- A reviewed member allowlist permits only required primitive raw-bit methods, selected
  `Math`/`StrictMath` exponential/logarithm operations, and typed [`MemorySegment`](../../../../glossary.md#term-memorysegment) access. Any
  unexpected generated member is a failure.
- Specialization/cache identity distinguishes kind, type, axis role, algorithm version/pass shape,
  access/carrier patterns, and every emitted-body fact. Runtime bases, offsets, compatible extents/
  strides, ranges, slots, buffers, workers, graph/run identities, and values remain cold unless a
  fact actually changes bytecode.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [Runtime glossary](../../../../glossary.md#term-runtime); [buffers glossary](../../../../glossary.md#term-buffer); [run glossary](../../../../glossary.md#term-run); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Schema advances exactly 46 to 47. Schema-46 and earlier envelopes safely miss; schema-42 retained
  evidence and resources remain immutable and explicitly historical.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Freeze eight bounded performance targets: the six dense heap-array cases for both kinds across
  all three types at Shape `[128, 2048]`, [axis](../../../../glossary.md#term-axis) one; one [FLOAT32](../../../../glossary.md#term-float32) `SOFTMAX` heap-to-segment case; and
  one FLOAT32 `LOG_SOFTMAX` segment-to-heap case using one frozen legal general layout. Freeze three
  unchanged-family controls for FLOAT32 `LOG_SUM_EXP`, ordinary `SUM`, and [pointwise](../../../../glossary.md#term-pointwise) [`EXP`](../../../../glossary.md#term-expx).
- Every [target](../../../../glossary.md#term-target) and control passes generated/direct `<= 1.15x` in each of five isolated forks and
  for the median of fork medians. The direct oracle performs identical useful and validation work,
  is frozen before timing, and is never slowed, padded, or routed through production helpers.<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median); [fork glossary](../../../../glossary.md#term-fork-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Focused tests, one final uncached CPU suite, Javadoc, retained semantic/Class-File/decompilation/
  member/checksum/five-fork evidence, exact scope, status, and whitespace gates pass. A distinct
  clean documentation context then finalizes Javadocs, CPU guide, glossary impact, planning status,
  and documentation validation without repeating stable Java/performance work.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [decompilation glossary](../../../../glossary.md#term-decompilation); [checksum glossary](../../../../glossary.md#term-checksum); [scope glossary](../../../../glossary.md#term-scope); [planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

The implementation context runs focused tests after stabilization, then one final [CPU](../../../../glossary.md#term-cpu) suite:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSoftmaxGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuSoftmaxInputValidatorTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuSoftmaxReferenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest
./gradlew :backends:cpu:test --rerun-tasks
```

Retain one [immutable](../../../../glossary.md#term-immutable) evidence bundle outside the repository containing exact source/probe inputs,
environment/[JVM](../../../../glossary.md#term-java-virtual-machine-jvm) facts, all eight [target](../../../../glossary.md#term-target) classes and three controls, specializations, full
`javap -c -p` and `javap -v -p`, member reports, semantic/accuracy/rejection/range/[canary](../../../../glossary.md#term-canary) results,
five isolated `-Xms1g -Xmx1g` [fork](../../../../glossary.md#term-fork-meanings) outputs, rejected whole [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), summaries, inventory, and a
[SHA-256](../../../../glossary.md#term-sha-256) [manifest](../../../../glossary.md#term-manifest). Each fork uses at least five randomized warmups, nine randomized measured rounds,
adaptive [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) of at least 25 ms, deterministic [finite](../../../../glossary.md#term-finite) inputs, randomized generated/direct
order, and raw/[checksum](../../../../glossary.md#term-checksum) verification. Reject and retain a whole fork if any target, control,
checksum, semantic, accuracy, environment, or admitted-domain gate fails.

The documentation-focused context receives the stabilized diff, final CPU XML, and retained
evidence and reuses successful Java/performance evidence unless executable behavior changes or a
concrete stale-evidence risk is recorded. After final Javadocs it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also inspects rendered Javadocs and validates Markdown links/anchors/fences/newlines/whitespace,
exact path/type ceilings, [schema](../../../../glossary.md#term-schema) 47 and historical schema-42 wording, CPU 0007D Complete, CPU
0007A1D Review needed, CPU 0007E Complete only after every gate, CPU 0007F+ Draft without detailed
specifications, and empty staging. Repository-wide, architecture, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and
integration suites remain deferred to CPU 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) because no shared boundary changes.

## Dependencies

- Complete Model 0016I owns exact first-class kinds, normalized-axis attributes, ideal
  mathematical meanings, and the explicit absence of a settled finite-precision/edge policy.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [axis glossary](../../../../glossary.md#term-axis); [finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision))</span>
- Complete Model 0016J owns FLOAT64/FLOAT32/BFLOAT16 expression eligibility, Shape/type/gradient
  preservation, one-input provenance, and explicit first-class construction.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [provenance glossary](../../../../glossary.md#provenance))</span>
- Complete Compiler 0005B and current inference preserve the kind, validate the floating
  shape-preserving occurrence, and construct gradients from the exact forward output without
  selecting a forward algorithm or numerical-edge rewrite.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Complete CPU 0007D supplies the schema-46 frontier, stable max-shift/compensated-sum precedent,
  mixed-carrier geometry, direct-emitter separation, Class-File/member inspection, and five-fork
  performance protocol. Its axis-removing IR is precedent, not the softmax IR owner.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [schema glossary](../../../../glossary.md#term-schema); [carrier glossary](../../../../glossary.md#term-storage-carrier); [fork glossary](../../../../glossary.md#term-fork-meanings); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [owner glossary](../../../../glossary.md#term-owner))</span>
- Existing shared Prepare/Runtime contracts can carry the two buffers and zero workspace without
  modification. CPU 0007A1D remains historical Review needed and is not a dependency.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [carry glossary](../../../../glossary.md#term-carry); [buffers glossary](../../../../glossary.md#term-buffer); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Follow-up tasks

- CPU 0007F remains the next ordered Draft task for layer, RMS, and batch normalization and
  receives no detailed specification here.<span data-glossary-navigation="true"> ([batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training))</span>
- A future Model/config/conformance decision may define portable NaN, infinity, zero-width, or
  relaxed numerical behavior; this task does not preempt or schedule that decision.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- CPU 0008B retains later explicit semantic-kernel recognition boundaries and must continue never
  to infer stable softmax from decomposed graphs.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [graphs glossary](../../../../glossary.md#graph))</span>
- [CPU](../../../../glossary.md#term-cpu) 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) retains repository-wide portable coverage, [conformance](../../../../glossary.md#term-conformance), and integration closure.
- CPU 0007A1D remains historical Review needed; its evidence gap is independent.

## Architecture impact

Expected impact: None.

This task adds a [CPU](../../../../glossary.md#term-cpu)-private generated forward family behind existing [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare),
and [Runtime](../../../../glossary.md#term-runtime) contracts. If implementation requires a public/shared numerical contract, [dependency](../../../../glossary.md#term-dependency),
[module](../../../../glossary.md#term-module), resource kind, architecture rule, or build change, stop and report rather than editing
that boundary.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007E. Work on the existing worktree
without committing, pushing, staging, resetting, reverting, deleting, or modifying unrelated
work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, CPU task 0007E and
its dependencies, Model 0016I/0016J, Compiler 0005B and current inference/gradient code, and the
current CPU capability/IR/lowering/emitter/reference/route/prepare/finalization/executable/cache
source and tests plus documentation rules and applicable profiles.

Implement exactly this specification within its 49-path/five-type ceilings. Preserve
first-class identity; never recognize decomposed graphs; enforce the CPU-private finite,
positive-width admitted subset before mutation/submission; add the five focused owners; advance
schema 46 to 47; and satisfy every semantic, accuracy, rejection, resource, Class-File, checksum,
control, and five-fork optimal-direct-Java gate without weakening or slowing the direct oracle.

Stop on any ceiling, architecture uncertainty, numerical-ownership conflict, or scope conflict.
Hand the stabilized diff, CPU XML, and retained evidence to a distinct clean documentation-focused
context to finalize documentation and planning in the same change. Do not mark Complete until all
gates pass. Do not commit, push, or stage.
```

## Local decisions

- The task uses a CPU-private fail-closed subset rather than treating unsettled edge policy as a
  blocker. Model and Compiler settle first-class identity, axis, Shape, and floating types but
  intentionally leave forward
  edge values open; rejecting non-finite inputs, non-finite shifts, and zero-width selected axes
  avoids inventing portable results while still delivering useful stable finite execution.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [identity glossary](../../../../glossary.md#term-identity-meanings); [axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [finite glossary](../../../../glossary.md#term-finite))</span>
- One emitter owns both kinds because they share maximum and normalization-sum passes and differ
  only in the generation-time final formula. A second emitter would duplicate one cohesive
  algorithm; a runtime kind switch is forbidden.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- One focused validator performs the complete admitted-domain scan before any worker submission.
  Folding it into each range would either repeat full validation or permit partial writes; adding
  it to the already broad executable would obscure ownership.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Three passes avoid output workspace and recompute exponentials for [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax). This matches the
  optimal clean Java oracle and keeps exact resource truth simpler than a saved-exponential buffer.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Binary64 intermediates are selected for FLOAT64/FLOAT32. BFLOAT16 uses the CPU plan's expected
  binary32 accumulation boundary and explicit binary32 narrowing points around transcendental
  calls. These are CPU-private stable policies, not public accumulation types or cross-backend
  bitwise promises.<span data-glossary-navigation="true"> ([Binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [narrowing glossary](../../../../glossary.md#term-narrowing); [backend glossary](../../../../glossary.md#backend))</span>
- Complete-[slice](../../../../glossary.md#slice) ranges make [scalar](../../../../glossary.md#term-scalar)/parallel results deterministic without partial/combine state.
- Schema 47 is the sole successor to 46 because one new generated family changes bytecode and
  compatibility. Skipping a number or retaining 46 would misidentify artifacts; no migration
  reader is warranted because older envelopes safely miss.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [bytecode glossary](../../../../glossary.md#term-bytecode); [artifacts glossary](../../../../glossary.md#term-artifact))</span>

## Known limitations

- Only fully static resolved-layout scalar/parallel-scalar execution over the CPU-private finite,
  positive-width subset is supported. Dynamic Shapes, zero-width selected axes, non-finite values,
  finite shift overflow, vectorization, partial/combine execution, fusion, and native routes remain
  unsupported.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [CPU glossary](../../../../glossary.md#term-cpu); [finite glossary](../../../../glossary.md#term-finite); [Shapes glossary](../../../../glossary.md#shape); [axes glossary](../../../../glossary.md#term-axis); [overflow glossary](../../../../glossary.md#term-overflow); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Finite results follow the selected CPU algorithm and tolerance table. Only generated/direct
  identity is raw-bit evidence; no cross-backend bitwise promise is added.<span data-glossary-navigation="true"> ([tolerance glossary](../../../../glossary.md#term-tolerance); [identity glossary](../../../../glossary.md#term-identity-meanings); [backend glossary](../../../../glossary.md#backend))</span>
- Softmax recomputes shifted exponentials in its store pass. Saved exponentials, workspace, and
  alternative autotuned algorithms remain deferred.<span data-glossary-navigation="true"> ([Softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- The performance gate covers only the frozen bounded matrix and environment; it is not a general
  tuning or throughput claim.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [throughput glossary](../../../../glossary.md#term-throughput))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation and review completed the exact [CPU](../../../../glossary.md#term-cpu)-private family specified above. The stabilized
worktree contains exactly five new production types: `CpuSoftmaxIr`, `CpuSoftmaxLowering`,
`CpuSoftmaxEmitter`, `CpuSoftmaxInputValidator`, and `CpuSoftmaxReferenceKernel`. Generated
compatibility advanced exactly once from [schema](../../../../glossary.md#term-schema) 46 to schema 47. No public or shared [API](../../../../glossary.md#term-application-programming-interface-api),
architecture, [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module), resource kind, build, [conformance](../../../../glossary.md#term-conformance), or integration contract
changed.

The implementation/review context supplied the final post-fix evidence. Its uncached
`./gradlew :backends:cpu:test --rerun-tasks` [run](../../../../glossary.md#term-run) passed 70 suites and 428 tests with zero failures
or errors and one expected opt-in skip. `./gradlew :backends:cpu:javadoc` passed with two expected
incubator-module warnings. No executable Java or [Javadoc](../../../../glossary.md#term-javadoc) changed after those commands, so clean
documentation-finalization context `01a03884-c233-7d72-b090-34721cff3b8b` reused that accepted
evidence and did not repeat Java tests or Javadoc.

Retained evidence at `/private/tmp/synaptik-cpu-0007e-evidence` contains 163 files. Its
`SHA256SUMS` lists 162 entries and `shasum -a 256 -c SHA256SUMS` verified all 162; the [manifest](../../../../glossary.md#term-manifest) is
the one additional file. The 70 retained XML files independently total 428 tests, zero failures,
zero errors, and one skip. The bundle contains eight generated [targets](../../../../glossary.md#term-target), three unchanged-family
controls, complete compatibility/[descriptors](../../../../glossary.md#term-descriptor), generated classes, `javap -c -p`, `javap -v -p`,
member reports, sources/probe classes, five accepted sequential isolated-[JVM fork](../../../../glossary.md#term-jvm-fork) results, one
pre-timing rejected launch, aggregate results, environment, inventory, and checksums.

All 55 target/control per-[fork](../../../../glossary.md#term-fork-meanings) ratios and all 11 [median](../../../../glossary.md#term-median)-of-fork-medians ratios passed the fixed
`<= 1.15x` gate. The worst individual result was `1.119471916x` for [BFLOAT16](../../../../glossary.md#term-bfloat16) LOG_SOFTMAX [heap](../../../../glossary.md#term-heap) in
fork 2; the worst aggregate was `1.113919290x` for the same case. The five accepted forks ran in
separate Java 26.0.1 processes with the [vector](../../../../glossary.md#term-vector) [incubator module](../../../../glossary.md#term-incubator-module) and fixed one-gibibyte heaps. Each
used five randomized warmups, nine randomized measured rounds, adaptive [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) of at least 25
milliseconds, deterministic [finite](../../../../glossary.md#term-finite) inputs, randomized generated/direct order, raw-bit result
verification, and an anti-elimination [checksum](../../../../glossary.md#term-checksum). The optimal clean Java comparisons performed the
same validation, traversal, algorithm, pass/[dataflow](../../../../glossary.md#term-dataflow) [shape](../../../../glossary.md#shape), Math work, [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout) addressing,
[narrowing](../../../../glossary.md#term-narrowing), and stores; they were not delayed or routed through production numerical helpers.

Complete generated-member inspection found zero `Math.log` references in [SOFTMAX](../../../../glossary.md#softmax--log-softmax) and retained the
required logarithm in LOG_SOFTMAX. All eleven retained classes are final, field-free,
constructor-free, and have exactly one static typed `invoke` entry. The audit found no forbidden
helper, allocation, boxing, reflection, [method handle](../../../../glossary.md#term-method-handle), `invokedynamic`, collection, string, or
semantic-[dispatch](../../../../glossary.md#term-dispatch) reference. Generated replay bytes and an independent recompilation of the 22
probe Class-Files were byte-identical.

Clean documentation-finalization context `01a03884-c233-7d72-b090-34721cff3b8b` applied General,
API/Javadoc, [Backend](../../../../glossary.md#backend) Guide, Developer Guide, [Planning](../../../../glossary.md#term-planning), and Example profiles. It read the governing
architecture and planning contracts, final source/Javadocs/package summaries, affected tests,
[Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) softmax contracts, CPU guide, glossary, complete task state, and retained evidence.
It finalized exactly the CPU guide, glossary, this task, CPU master plan, and roadmap. Production
Javadocs/package summaries were already accurate and complete, including [parameters](../../../../glossary.md#term-parameter), results,
failures, [lifecycle](../../../../glossary.md#lifecycle) boundaries, first-class [identity](../../../../glossary.md#term-identity-meanings), admitted-domain validation, direct
three-pass algorithms, exact Math behavior, resources, and schema identity, so no Java path was
changed. Documentation checks validated local Markdown links and heading anchors, balanced
fences, final newlines, terminology, the `[1, 2, 3]` numerical example, exact five-document [scope](../../../../glossary.md#term-scope),
the 49-path ceiling, exactly five new production types, schema/status history, hidden untracked
files, unstaged index, `git diff --check`, and `git diff --cached --check`.

Reasoned no-change conclusions:

- `ARCHITECTURE.md`, the current architecture explanations, ADRs, and architecture tests remain
  accurate because the implementation stays inside existing concrete-backend analysis,
  finalization, generated-kernel, cold-binding, and execution ownership and changes no dependency
  or module boundary.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [binding glossary](../../../../glossary.md#term-binding); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Backend-conformance and integration tests remain unchanged because this task implements a
  CPU-private admitted subset and makes no cross-backend or end-to-end Engine promise. Their
  broader checkpoint remains CPU 0009 or continuous integration.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci))</span>
- Model semantic kinds, Tensor expressions, Compiler inference and exact-forward-output gradient
  rules, and their public API documentation remain accurate because CPU preserves the explicit
  first-class occurrence and adds no Model, Compiler, decomposition, gradient, or public
  numerical-edge contract. Compile and Training APIs therefore require no change.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient); [occurrence glossary](../../../../glossary.md#term-occurrence); [decomposition glossary](../../../../glossary.md#term-decomposition); [Compile glossary](../../../../glossary.md#compile))</span>
- Runtime, shared Prepare, backend-contract, and Trace contracts remain accurate because the
  family uses their existing two-buffer, zero-workspace, staged preparation, cold binding, and
  prepared-execution boundaries without new shared state or dispatch.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Root/CPU Gradle and Java 26 configuration remain accurate because no dependency, project,
  source set, toolchain, plugin, task, incubator requirement, or build behavior changed.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [plugin glossary](../../../../glossary.md#term-plugin))</span>
- Unrelated guides, APIs, operation families, historical schema-42 evidence, and CPU 0007A1D
  remain accurate because this task neither edits nor reinterprets their behavior or evidence.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

## Implementation notes

- Added one exact capability/lowering path for explicit first-class SOFTMAX and LOG_SOFTMAX over
  matching FLOAT64/FLOAT32/BFLOAT16 descriptors, fully static resolved legal layouts, positive
  selected width, arbitrary non-selected zero extents, and an injective distinct output.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [descriptors glossary](../../../../glossary.md#term-descriptor); [layouts glossary](../../../../glossary.md#layout); [injective glossary](../../../../glossary.md#term-injective))</span>
- Added the five focused CPU-private owners and integrated their first-class IR, complete-slice
  geometry, direct typed generation, pre-write finite-domain validation, and independent clean
  Java oracle through existing route, preparation, finalization, execution, and cache seams.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [slice glossary](../../../../glossary.md#slice); [finite glossary](../../../../glossary.md#term-finite); [validation glossary](../../../../glossary.md#term-validation); [oracle glossary](../../../../glossary.md#term-test-oracle); [cache glossary](../../../../glossary.md#term-cache))</span>
- Added direct stable three-pass bodies. SOFTMAX finds the maximum, accumulates the compensated
  shifted-exponential sum, then reevaluates exponentials and divides; it does not calculate
  `log(sum)`. LOG_SOFTMAX finds the same maximum and sum, calculates one logarithm per slice, then
  stores shift minus logarithm without another exponential.
- Added focused and integration tests for exact capability, explicit-kind preservation, geometry,
  carrier/layout combinations, empty no-op behavior, finite-domain and overlap rejection before
  writes/submission, raw-bit generated/reference identity, resource truth, deterministic ranges,
  schema/cache identity, and generated class/member shape.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [reference identity glossary](../../../../glossary.md#term-reference-identity); [schema glossary](../../../../glossary.md#term-schema); [shape glossary](../../../../glossary.md#shape))</span>
- Advanced current generated compatibility exactly from schema 46 to schema 47. Schema-46 and
  earlier artifacts are safe misses; retained schema-42 material remains historical.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Finalized the CPU backend guide, glossary, task, CPU master plan, and roadmap without changing
  executable Java or Javadocs during the independent documentation pass.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>

## Completion summary

- Completed changes: Added and documented the explicit stable SOFTMAX/LOG_SOFTMAX portable CPU
  family over its finite, positive-selected-width admitted subset, with direct typed generated
  execution, complete pre-write validation, arbitrary legal resolved layouts/carriers,
  complete-slice parallelism, and zero workspace.<span data-glossary-navigation="true"> ([SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [CPU glossary](../../../../glossary.md#term-cpu); [finite glossary](../../../../glossary.md#term-finite); [validation glossary](../../../../glossary.md#term-validation); [layouts glossary](../../../../glossary.md#layout); [slice glossary](../../../../glossary.md#slice))</span>
- Files changed or created: 47 total paths—29 production/Javadoc paths, 13 test paths, and exactly
  five documentation/planning paths—including exactly five new CPU-private production types.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: The accepted post-fix uncached CPU run passed 70 suites/428 tests with one
  expected skip and no failures/errors; CPU Javadoc passed with two incubator warnings; all 55
  fork and 11 aggregate performance ratios passed; retained semantic/Class-File/member/checksum
  evidence passed; final Markdown/link/anchor/fence/terminology/status/scope/staging/whitespace
  checks passed.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [fork glossary](../../../../glossary.md#term-fork-meanings); [checksum glossary](../../../../glossary.md#term-checksum); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Clean context `01a03884-c233-7d72-b090-34721cff3b8b` completed the
  required independent pass without rerunning stable Java or performance work.
- Documentation impact: CPU guide and glossary now explain the exact current capability,
  CPU-private numerical boundary, algorithms, validation, resources, schema, and bounded evidence;
  task, master plan, and roadmap are synchronized to Complete.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Javadoc review: All affected production Javadocs and package summaries remain complete and
  accurate unchanged; no Java path changed in the documentation pass.
- Glossary impact: The existing softmax/log-softmax entry now distinguishes current Model and
  Compiler meaning from the CPU-private executable subset and avoids cross-backend promises.<span data-glossary-navigation="true"> ([log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for CPU 0007E. CPU 0007A1D remains Review needed; CPU 0007F remains the
  next ordered Draft task without a detailed specification.

Status: Complete
