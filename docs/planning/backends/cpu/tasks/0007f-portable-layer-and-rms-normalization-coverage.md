# Task 0007F: Portable Layer and RMS Normalization Coverage

## Status

Complete

## Goal

Execute exactly one fully static, resolved-layout, first-class `LAYER_NORM` or `RMS_NORM`
occurrence on the portable CPU route. Cover every current no-affine, affine, no-scale, and scaled
signature with direct deterministic trailing-slice generated loops whose semantic algorithm,
pass/dataflow shape, and avoidable-overhead profile match optimal clean Java implementations.

## Mental model

```text
one explicit layer- or RMS-normalization node
  -> CPU validates one static trailing-Shape geometry
  -> CPU retains exact ordered semantic inputs and result type
  -> one generation-time numerical body
       layer: exact mean -> corrected centered square sum -> standardize/affine
       RMS:   scaled square sum -> normalize/scale
  -> scalar or complete-slice parallel-scalar execution
```

The operation remains its first-class Model kind through lowering. CPU never recognizes a
decomposed arithmetic/reduction graph as layer or RMS normalization.

## Scope

- Recognize exactly one compiled `LayerNormKind.LAYER_NORM` occurrence with either:
  - `LayerNormAttrs` and ordered `[input] -> [output]`; or
  - `AffineLayerNormAttrs` and ordered `[input, scale, bias] -> [output]`.
- Recognize exactly one compiled `RmsNormKind.RMS_NORM` occurrence with `RmsNormAttrs` and either:
  - ordered `[input] -> [output]`; or
  - ordered `[input, scale] -> [output]`.
- Preserve first-class semantic identity. Admit only the explicit kinds and exact attribute/input
  cardinality pairings; never infer, fuse, reconstruct, or relabel normalization from decomposed
  graphs.
- Accept every current BFLOAT16, FLOAT32, and FLOAT64 input combination. No-affine/no-scale forms
  retain the input type. Affine Layer Norm promotes `input`, `scale`, then `bias` in occurrence
  order; scaled RMS Norm promotes `input`, then `scale`. Require output and typed epsilon to equal
  that exact result type, and require output gradient eligibility to equal the Model-defined input
  eligibility or ordered-input logical OR. Promotion is generated conversion at reads and final
  stores, not inserted Model casts.
- Require a fully static rank-positive input Shape and non-empty static `normalizedShape` that
  exactly equals the trailing input Shape. Require scale and bias Shape, when present, to equal
  `normalizedShape`. Require fully resolved non-negative layouts, checked representable spans, and
  an injective writable output with the exact input Shape.
- Treat any zero leading or normalized extent as an empty output. Perform no divisor, statistic,
  input read, output write, workspace use, generated invocation, or worker submission. Otherwise
  require the checked normalized-domain count and leading slice count to be positive `long`
  values before resource declaration.
- Traverse leading slices in canonical logical row-major order and each normalized trailing slice
  in canonical logical row-major order. Reuse scale/bias normalized coordinates for every leading
  slice. Flattened generated `start`/`end` bounds count complete leading slices and never divide a
  normalized slice.
- Layer Norm uses population correction zero and epsilon inside the square root:

  ```text
  mean           = sum_i(x_i) / N
  variance       = sum_i((x_i - mean) * (x_i - mean)) / N
  standardized_i = (x_i - mean) / sqrt(variance + epsilon)
  output_i       = standardized_i                         // no affine
  output_i       = standardized_i * scale_i + bias_i     // affine
  ```

- Implement finite nonconstant Layer Norm with three direct passes. Pass one classifies the slice,
  detects a finite constant slice, and uses the existing exact-sum generation owner plus one final
  result-computation-format rounding to obtain `mean`. Pass two accumulates deviations and squared
  deviations using primitive compensated binary64 sums, applies
  `sumSquares - sumDeviation * sumDeviation / N`, clamps only a negative roundoff residue to
  positive zero, and divides by `N`. Pass three standardizes and applies the optional affine
  multiply/add in the result type.
- Preserve Layer Norm special classes exactly: any NaN or either infinity in the input slice makes
  every standardized value NaN and affine arithmetic cannot suppress it; every finite constant
  slice, including mixed signed zeros, produces exact positive-zero standardized values before
  affine arithmetic; all other overflow, underflow, signed-zero, infinity, and NaN outcomes follow
  the specified result-format operations. Do not add a variance clamp beyond the negative
  roundoff-residue correction above.
- RMS Norm uses uncentered population mean square and epsilon inside the square root:

  ```text
  meanSquare  = sum_i(x_i * x_i) / N
  rms         = sqrt(meanSquare + epsilon)
  normalized_i = x_i / rms
  output_i    = normalized_i                    // no scale
  output_i    = normalized_i * scale_i          // scaled
  ```

- Implement RMS Norm with two direct passes. Pass one classifies the complete slice and uses the
  standard scaled sum-of-squares state `(scale, scaledSquares)` to avoid avoidable finite
  overflow/underflow before deriving the population mean square and root. Pass two divides each
  promoted input by that root and performs the optional result-type scale multiplication.
- Preserve RMS special classes exactly: any NaN makes every normalized value NaN; without NaN, any
  input infinity makes the root positive infinity, finite inputs produce same-signed zero, and
  infinite inputs produce NaN; all-finite zero inputs preserve their sign before optional scale;
  finite overflow to a positive-infinite root produces same-signed zero numerators; optional scale
  follows ordinary result-format NaN, infinity, and signed-zero multiplication.
- For both families, treat input values as the result type before numerical work. BFLOAT16 and
  FLOAT32 results use at least FLOAT32 computation and FLOAT64 results use FLOAT64 computation.
  Reuse CPU 0007D's equal-or-wider exact-sum, compensated-sum, and scaled-square precedents, narrow
  the Layer mean to FLOAT32 for BFLOAT16/FLOAT32 results or FLOAT64 for FLOAT64, and evaluate the
  final division plus optional multiply/add in FLOAT32 for BFLOAT16/FLOAT32 or FLOAT64 for
  FLOAT64. Do not fuse the affine multiply/add. Encode BFLOAT16 once at the final output store;
  FLOAT32/FLOAT64 use their ordinary primitive operation boundaries. Make generated/direct
  traversal and rounding identical. This equal-or-wider BFLOAT16 policy is a CPU-private
  conforming reassociation, not a public promotion or cross-backend bitwise promise.
- Declare ordered logical inputs exactly as captured. If multiple semantic input positions refer
  to the same exact logical value, declare that boundary once in first-occurrence order and retain
  an immutable position-to-boundary map; do not duplicate physical resources. Declare exactly one
  output. Input/input alias is harmless read sharing; reject every output/input overlap before
  mutation or submission.
- Support each boundary's matching typed heap array or native-order `MemorySegment` carrier:
  `short[]`, `float[]`, or `double[]` according to its own data type. Support every heap/segment
  combination for the unique ordered boundaries and output, arbitrary legal offsets and positive
  or zero input strides, and arbitrary injective output strides. Generated hot work uses concrete
  typed entries after one cold carrier binding and contains no storage discovery or carrier switch.
- Use primitive locals for classification, compensated sums, scaled squares, roots, coordinates,
  counts, and addresses. Layer Norm alone may use the existing alignment-eight
  `AGGREGATE_EXACT_STATE` workspace for the exact-mean pass: exactly one disjoint maximum-size
  slice per simultaneously used range. RMS Norm uses zero workspace. Add no resource kind,
  materialization, partial value, combine state, saved statistic, or hidden scratch.
- Reuse existing bounded scalar/parallel-scalar orchestration. Declare no worker, executor,
  thread-local, or scheduling resource: worker count and `start`/`end` ranges remain cold
  invocation facts, and generated code creates or submits no workers. Exact-state slice count is
  derived from the maximum number of simultaneously executing Layer ranges, not requested worker
  count or a hidden thread-local assumption.
- Add one focused trailing-normalization IR and one focused trailing-Shape lowering owner. Add
  separate Layer Norm and RMS Norm emitters and one independent reference owner. Reuse existing
  CPU-private access, exact-sum emission, carrier, loop, route, preparation, binding, and artifact
  seams without adding either kind to aggregate/softmax IR or growing a god emitter.
- Emit one deterministic final, field-free, constructor-free generated class per specialization
  with one public static typed `invoke` entry. Advance `CpuGeneratorSchema.CURRENT_VERSION`
  exactly once from 47 to 48 because the new family, multi-input carrier identity, numerical
  bodies, and workspace facts change artifact meaning. Schema 47 and earlier artifacts are safe
  incompatible misses; retained historical evidence remains immutable.
- Use well-written optimal clean Java loops with identical validation work, pass count, traversal,
  exact-state/scaled-square arithmetic, carrier/layout addressing, conversions, and stores as the
  generated/direct oracle. Generated code may call only required JDK primitive/raw-bit,
  `Math`/`StrictMath`, and typed `MemorySegment` operations and must call no Synaptik numerical,
  validation, or reference helper.

## Out of scope

- `BATCH_NORM_INFERENCE`, which belongs to Draft CPU 0007F1, and `BATCH_NORM_TRAINING`, which
  belongs to Draft CPU 0007F2. This task has no channel-axis geometry, running statistics,
  momentum, saved statistics, statistic transitions, or five-output binding.
- Training/evaluation flags, mutable state, parameter/layer ownership, backward execution, saved
  means/inverse standard deviations, gradient-policy changes, or special execution for
  compiler-generated gradients. Layer/RMS forward occurrences have no mode distinction; ordinary
  compiler-generated gradient graphs remain ordinary graph work.
- Group, instance, local-response, weight, spectral, partial RMS, or other normalization; bias in
  RMS Norm; bias-only Layer Norm; broadcast scale/bias; default or Tensor epsilon; configurable
  correction, accumulator, approximation, clamp, sentinel, or relaxed-math policy.
- Decomposition of a first-class kind; recognition or fusion of decomposed graphs; fusion with
  adjacent work; native/vendor or Vector API routes; runtime kind/signature dispatch.
- Dynamic or symbolic Shapes, unresolved layouts, runtime normalized axes/Shapes, scalar inputs,
  FLOAT16, integral/BOOL/quantized/complex data, a count/span that fails checked representation,
  output aliasing, negative resolved strides, or non-injective output layouts.
- Partial normalized slices, partial reductions, combine trees, per-element tasks, workspace for
  RMS Norm, materialized affine operands, saved standardized values, or new resource kinds.
- Generated calls to CPU validation/reference implementations or Synaptik arithmetic helpers;
  allocation, boxing, reflection, method handles, `invokedynamic`, generic `Object` descriptors,
  collection/map/string lookup, or avoidable virtual/semantic dispatch in generated hot work.
- Public/shared API, Model, Compiler, Training, Runtime, Prepare contract, backend-contract,
  dependency, module-boundary, Gradle/toolchain, architecture, conformance, integration, tracing,
  tuning, or persistence-format changes beyond the CPU-private generator schema.
- Reopening CPU 0007A1D or creating detailed specifications for CPU 0007F1 or any later task.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007D advanced reductions`](0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md)
- [`CPU 0007E stable softmax`](0007e-portable-stable-softmax-and-log-softmax-coverage.md)
- [`Model 0021 Layer Norm`](../../../modules/model/tasks/0021-layer-normalization-semantics-and-tensor-expressions.md)
- [`Model 0021A RMS Norm`](../../../modules/model/tasks/0021a-rms-normalization-semantics-and-tensor-expressions.md)
- [`Model 0021B batch inference`](../../../modules/model/tasks/0021b-batch-normalization-inference.md)
- [`Model 0021C batch training`](../../../modules/model/tasks/0021c-batch-normalization-training-and-statistic-transition.md)
- [`Compiler 0005B normalization inference and gradients`](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [`Prepare 0001 resource declaration`](../../../modules/prepare/tasks/0001-backend-partition-analysis-and-resource-declaration.md)
- [`Prepare 0002 finalization handoff`](../../../modules/prepare/tasks/0002-backend-partition-finalization-handoff.md)

## Architecture constraints

- Model owns first-class kinds, exact signatures, trailing normalized Shape, formulas, epsilon,
  promotion, Shape, gradient eligibility, and special-value classes. Compiler owns capture,
  inference/revalidation, and gradients. CPU owns only truthful forward admission, static lowering,
  concrete numerical algorithm, access/resource validation, and generated realization.
- CPU analysis must select only explicit Layer/RMS occurrences. Neither CPU nor Compiler may infer
  first-class normalization from an equivalent-looking decomposed graph.
- The existing Model and Compiler contracts are complete for this task: all four admitted forms,
  result descriptors, typed epsilon rules, finite targets, special classes, and gradient formulas
  are settled. Implementation must stop rather than revise those contracts.
- CPU analysis/lowering computes all static eligibility and exact resources before shared slot
  assignment. CPU executable binding validates carriers, thread access, spans, alignment,
  injectivity, and overlap before mutation or worker submission. Shared Prepare remains CPU-blind;
  Runtime invokes the cold-bound recipe without operation dispatch.
- Generated code must preserve the optimal clean Java semantic algorithm, hot-loop/pass/dataflow
  shape, and avoidable-overhead profile. A generic bridge, reference call, per-element callback,
  packed runtime kind switch, or decomposed realization is not a specialized implementation.
- Any needed public/shared/build/architecture/conformance/integration change, new resource kind,
  materialization, partial/combine protocol, unresolved semantic decision, or batch-normalization
  implementation is a stop/replan condition.

## Package impact

Existing CPU-private packages changed:

- `io.github.pho001.synaptik.backend.cpu` — exact capability admission.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — immutable trailing-normalization identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — static trailing-slice geometry and
  exact resource derivation.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — separate direct Layer/RMS bodies.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent differential oracle.
- existing `route.portable`, `prepare`, `executable`, and `cache` packages — realization,
  declaration, cold validation/invocation, and schema-48 compatibility.

Packages added: None.

Type placement:

- Add `CpuTrailingNormalizationIr` in `internal.ir` for exact kind/signature, each boundary type,
  typed epsilon raw bits, normalized rank, algorithm version, pass/resource identity, access
  patterns, and structural specialization facts.
- Add `CpuTrailingNormalizationLowering` in `internal.lowering` for exact Shape/layout, leading
  slice count, normalized-domain count, canonical coordinate mapping, first-occurrence unique
  boundary mapping, checked spans, and exact Layer-only workspace derivation.
- Add `CpuLayerNormEmitter` and `CpuRmsNormEmitter` in `internal.codegen.emit`; each owns only its
  direct numerical body and is selected at generation time without runtime semantic dispatch.
- Add `CpuTrailingNormalizationReferenceKernel` in `internal.reference`; it independently derives
  coordinates and mathematical expectations without production lowering, emission, binding, or
  generated helpers.
- Add no validator type because every represented input value has defined Model behavior and the
  existing cold structural validation is sufficient. Add no facade, manager, registry, generic
  normalization interpreter, or aggregate/softmax owner change.

## Affected files

Expected production/Javadoc paths:

- `CpuCapabilityProvider.java` and CPU package Javadoc;
- `internal/ir/CpuPortableKernelIr.java`, new `CpuTrailingNormalizationIr.java`, and IR package
  Javadoc;
- `internal/lowering/CpuPartitionLowering.java`, new `CpuTrailingNormalizationLowering.java`, and
  lowering package Javadoc;
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new `CpuLayerNormEmitter.java`, new
  `CpuRmsNormEmitter.java`, the existing exact-sum/carrier/loop emission seams only where required,
  and emitter package Javadoc;
- `internal/cache/CpuGeneratorSchema.java`, `CpuKernelSpecialization.java` only if exact emitted
  identity requires it, and cache package Javadoc;
- `internal/route/portable/CpuPortableRoutePlan.java` and package Javadoc;
- `internal/prepare/CpuPartitionPreparationPlan.java`, `CpuPartitionPreparer.java`,
  `CpuPartitionFinalizer.java`, and package Javadoc;
- `internal/executable/CpuPreparedExecutable.java` and executable package Javadoc; and
- new `internal/reference/CpuTrailingNormalizationReferenceKernel.java`, the minimal existing
  differential seam if required, and reference package Javadoc.

Expected test paths:

- capability and internal-package inventory tests;
- new `CpuTrailingNormalizationIrTest`, `CpuTrailingNormalizationLoweringTest`,
  `CpuLayerNormGeneratedKernelTest`, `CpuRmsNormGeneratedKernelTest`, and
  `CpuTrailingNormalizationReferenceTest`;
- generated-class shape/member, specialization/cache, portable route, prepare/finalization,
  executable, and reference-differential tests only where their exact inventories or behavior
  change; and
- the historical pointwise-ledger test only if a current-schema assertion requires it; its
  historical evidence resource must not change.

Expected documentation/planning paths:

- `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  global roadmap. Architecture, public Tensor/Compile/Training APIs, Model/Compiler plans,
  shared Prepare, backend-contract, conformance, integration, and build files are review-only
  unless a stop condition is reached.

## Maximum scope

The complete implementation, tests, and documentation may modify or create at most 49 repository
paths: at most 29 production/Javadoc paths, 15 test paths, and exactly the five named documentation
and planning paths. Exactly five new CPU-private production types are permitted: the IR, lowerer,
two numerical emitters, and independent reference owner listed above.

This exceeds the planning guide's ordinary file-count guardrail only because one truthful CPU
family must update existing capability, sealed IR, generator, schema/cache, route, prepare,
executable, package-Javadoc, and inventory/differential seams atomically. The behavioral core is
still five focused new types in one module. Splitting those mechanical integrations from the IR
and emitters would temporarily advertise, persist, or prepare a family that cannot execute and
would duplicate final module/evidence validation; Layer/RMS themselves share the exact
trailing-slice/resource boundary that makes this one cohesive implementation session.

A 50th path, a sixth production type, another numerical emitter/validator, a generic normalization
owner, an aggregate/softmax semantic change, a public/shared/build/architecture/conformance/
integration path, RMS workspace, materialization, or partial/combine state is a stop/replan
condition. An omitted expected path does not authorize an unrelated one.

## Acceptance criteria

- Capability admits exactly the four explicit forms and full current BFLOAT16/FLOAT32/FLOAT64
  promotion matrix, with exact attributes, ordered input counts, static Shapes, result descriptor,
  epsilon type, eligibility, resolved layouts, checked spans, and injective output. Every excluded
  occurrence fails closed.
- Source-backed tests construct equivalent decomposed arithmetic/reduction graphs and prove they
  remain ordinary nodes and never become `CpuTrailingNormalizationIr`.
- Lowering records exact kind/form, every operand/result type, normalized Shape/rank/count, leading
  slice count, position-to-unique-boundary map, logical traversal, access/carrier facts,
  algorithm/pass identity, and exact Layer-only workspace. Checked count/address arithmetic
  precedes resource declaration.
- Semantic/differential coverage includes all four forms and all three result types; same-type and
  mixed floating operands; normalized rank one and higher; rank-equal and leading dimensions;
  normalized extent one and larger; leading/normalized zero extents; dense/general/zero-stride
  inputs; non-contiguous injective outputs; offsets; repeated logical inputs where Shape permits;
  and representative all-heap, all-segment, and mixed multi-boundary carrier patterns.
- Layer tests cover population variance, epsilon inside root, affine reuse, finite constants,
  mixed signed zeros, cancellation, subnormals, raw NaNs, both infinities, finite accumulator
  overflow/underflow, NaN scale/bias, infinite scale/bias, and ordinary affine sign/class rules.
- RMS tests cover uncentered mean square, epsilon inside root, scaled operand reuse, zeros of both
  signs, nonzero finite constants, subnormals, raw NaNs, either infinity, mixed finite/infinite
  slices, finite square/sum overflow pressure, and NaN/infinite/zero scale behavior.
- Mathematical accuracy uses an independent high-precision or StrictMath-based oracle and a frozen
  table: FLOAT64 Layer results are within 16 result ulps and RMS results within 8 result ulps;
  FLOAT32 results are within 4 result ulps; BFLOAT16 results are within one represented BFLOAT16
  ulp. Exact special-value classes, infinities, and zero signs use class/raw-sign assertions rather
  than tolerance. Generated/direct results for the identical algorithm are raw-bit equal.
- Resource declarations list unique logical inputs in first-occurrence order and exactly one
  output. Layer Norm alone declares one existing alignment-eight `AGGREGATE_EXACT_STATE` slice per
  simultaneously used range, sized from the exact maximum admitted normalized domain. RMS Norm
  declares zero workspace. Neither declares materialization, partial values, combine state, saved
  statistics, or hidden scratch.
- Empty outputs retain truthful existing boundary declarations but do no input read, output write,
  scratch use, generated call, or worker submission.
- Cold execution validates concrete carrier compatibility, liveness/thread access, native order,
  alignment, full accessed spans, output writability/injectivity, output/input overlap, and
  workspace/buffer overlap before mutation/submission. Failures leave output, scratch, inputs, and
  canaries unchanged and submit no work.
- Scalar and parallel-scalar results are raw-bit deterministic across legal complete-slice ranges,
  worker counts, repeated runs, and concurrent runs. No range divides a normalized slice and every
  simultaneous Layer range receives disjoint scratch.
- Generated classes are deterministic, final, field-free, constructor-free, and expose one typed
  static `invoke` with concrete unique input/output carriers, optional Layer scratch, packed
  primitive geometry, and `long start/end`; no `Object`, bridge descriptor, kind argument, or
  validation/reference call exists.
- Complete `javap -c -p` and `javap -v -p` inspection proves direct pass counts, inline typed
  carrier/address work, one final store per output position, generation-time family/form/type
  specialization, scale/bias reuse, and absence of allocation, boxing, reflection, method handles,
  `invokedynamic`, collections/maps/strings, semantic dispatch, and avoidable coordinate/address
  recomputation.
- A reviewed complete member allowlist permits only required primitive/raw-bit operations,
  `Math`/`StrictMath.sqrt`, and typed `MemorySegment` access. Any unexpected generated member is a
  failure; generated classes reference no Synaptik numerical/reference helper.
- Specialization/cache identity distinguishes kind, exact signature/form, every boundary/result
  type, exact typed epsilon raw bits, normalized-rank and code-shaping geometry, algorithm/pass
  version, exact-state requirement, and carrier/access patterns. Epsilon is an emitted constant,
  so two occurrences with different exact epsilon bits are incompatible specializations. Runtime
  bases, offsets, compatible extents/strides, ranges, slots, buffers, workers, graph/run identities,
  and Tensor values remain cold unless they change emitted bytes.
- Schema advances exactly 47 to 48. Schema-47 and earlier envelopes safely miss; all earlier
  retained evidence/resources remain immutable and explicitly historical.
- Freeze fourteen bounded performance targets over Shape `[128, 2048]` and normalized Shape
  `[2048]`: all twelve dense heap-array family/form/result-type cases (no-affine and affine Layer,
  no-scale and scaled RMS across three result types), one mixed-type/mixed-carrier affine Layer
  case, and one mixed-type/mixed-carrier scaled RMS case with frozen legal general layouts. Freeze
  three unchanged-family controls for FLOAT32 `VARIANCE`, `SOFTMAX`, and pointwise `ADD`.
- Every target and control passes generated/direct `<= 1.15x` in each of five isolated forks and
  for the median of fork medians. The optimal clean Java oracle performs identical validation,
  useful arithmetic, pass/dataflow shape, address work, conversions, and stores; it is frozen
  before timing and is never slowed, padded, or routed through production helpers.
- Focused tests, one final uncached CPU suite, Javadoc, retained semantic/Class-File/decompilation/
  member/checksum/five-fork evidence, exact scope, status, and whitespace gates pass. A distinct
  clean documentation context then finalizes affected Javadocs, CPU guide, glossary impact, task,
  master plan, and roadmap without repeating stable Java/performance work.

## Tests / validation

The implementation context runs focused tests after stabilization, then one final CPU suite:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuLayerNormGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuRmsNormGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuTrailingNormalizationReferenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest
./gradlew :backends:cpu:test --rerun-tasks
```

Retain one immutable evidence bundle outside the repository containing exact source/probe inputs,
environment/JVM facts, all fourteen targets and three controls, specializations, full
`javap -c -p` and `javap -v -p`, complete member reports, semantic/accuracy/range/alias/canary
results, five isolated `-Xms1g -Xmx1g` fork outputs, rejected whole samples, summaries, inventory,
and a SHA-256 manifest. Each fork uses at least five randomized warmups, nine randomized measured
rounds, adaptive batches of at least 25 ms, deterministic inputs, randomized generated/direct
order, and raw/checksum verification. Reject and retain a whole fork if any target, control,
checksum, semantic, accuracy, environment, resource, or scope gate fails.

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
exact path/type ceilings, schema 48 and historical-evidence wording, CPU 0007A1D Review needed,
CPU 0007D/0007E Complete, CPU 0007F Complete only after every gate, CPU 0007F1 and all later tasks
Draft without detailed specifications, and empty staging. Repository-wide, architecture,
backend-conformance, and integration suites remain deferred to CPU 0009 or CI because no shared
boundary changes.

Validation tiers are therefore explicit: focused tests during implementation plus one final
uncached CPU module suite are task validation; the clean documentation/Javadoc pass closes the
same task; repository-wide, architecture, backend-conformance, and integration validation is the
CPU 0009 capability-checkpoint or CI tier unless implementation reaches a stop condition that
changes a shared boundary.

## Dependencies

- Complete Model 0021 owns Layer Norm's two exact signatures, trailing Shape, population formula,
  affine rules, promotion, epsilon, metadata, and special-value classes.
- Complete Model 0021A owns RMS Norm's exact ranged signature, trailing Shape, uncentered formula,
  optional scale, promotion, epsilon, metadata, and special-value classes.
- Complete Model 0021B/0021C prove batch inference and batch training are distinct semantic and
  lifecycle families; they justify CPU 0007F1/0007F2 ordering but are not implementation inputs to
  this task.
- Complete Compiler 0005B and current `ReductionNormalizationInference`/
  `NormalizationGradientRules` own exact normalization signature revalidation and gradients. They
  preserve first-class Layer/RMS occurrences and require no forward CPU contract change.
- Complete CPU 0007D supplies exact floating-sum workspace, corrected two-pass statistics,
  scaled-sum-of-squares, mixed-carrier, direct-emitter, Class-File/member, and five-fork evidence
  precedents. Complete CPU 0007E supplies the schema-47 frontier and complete-slice multi-pass
  execution precedent. Neither existing IR owns normalization.
- Existing shared Prepare/Runtime contracts can carry the unique input/output boundaries and
  optional existing exact-state workspace unchanged. CPU 0007A1D remains historical Review needed
  and is not a dependency.

All dependencies and semantics are settled; there is no architectural or semantic blocker to
implementation within this task's fail-closed static scope.

## Follow-up tasks

- CPU 0007F1 remains Draft for stateless, coordinatewise, five-input/one-output
  `BATCH_NORM_INFERENCE` with arbitrary normalized channel axis and zero reduction workspace. It
  receives no detailed specification here.
- CPU 0007F2 remains Draft after 0007F1 for five-input/five-output `BATCH_NORM_TRAINING`, including
  non-channel reductions, biased forward variance, unbiased running-variance transition, momentum,
  saved statistics, mixed output Shapes, workspace, and multi-output binding. It receives no
  detailed specification here.
- CPU 0008 follows CPU 0007F2 for the remaining heavy portable families. CPU 0008B must preserve
  the prohibition on recognizing decomposed normalization as a first-class semantic kernel.
- CPU 0009 or CI retains repository-wide portable coverage, conformance, and integration closure.
- CPU 0007A1D remains historical Review needed; its evidence gap is independent.

## Architecture impact

Expected impact: None.

This task adds a CPU-private generated forward family behind unchanged Model, Compiler, Prepare,
and Runtime contracts. If implementation requires a public/shared numerical contract, dependency,
module, resource kind, architecture rule, or build change, stop and report rather than editing
that boundary.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007F. Work on the existing worktree
without committing, pushing, staging, resetting, reverting, deleting, or modifying unrelated
work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, CPU task 0007F and
its dependencies, Model 0021/0021A, the relevant Model normalization source contracts, Compiler
0005B and current normalization inference/gradient code, CPU 0007D/0007E, and current CPU
capability/IR/lowering/emitter/reference/route/prepare/finalization/executable/cache source and
tests plus documentation rules and applicable profiles.

Implement exactly CPU 0007F within its 49-path/five-type ceilings. Admit only explicit first-class
Layer Norm and RMS Norm with all four current forms and exact ordered floating promotion; never
recognize decomposed graphs. Add the shared trailing-normalization IR/lowerer, separate direct
Layer/RMS emitters, and independent reference owner. Preserve exact special classes, complete-
slice deterministic ranges, Layer-only existing exact-state scratch, RMS zero workspace, cold
validation before mutation/submission, and schema 47-to-48 identity. Directly generated bytecode
must match the semantic algorithm, loop/dataflow shape, and avoidable-overhead profile of the
frozen optimal clean Java implementation. Satisfy every semantic, accuracy, resource, alias,
Class-File/member, checksum, control, and five-fork performance gate without weakening or slowing
the oracle.

Stop on any ceiling, architecture uncertainty, semantic conflict, resource change, batch-
normalization implementation, or scope conflict. Hand the stabilized diff, final CPU XML, and
retained evidence to a distinct clean documentation-focused context to finalize Javadocs, CPU
guide, glossary, and planning in the same overall change. Do not mark Complete until all gates
pass. Do not commit, push, or stage.
```

## Local decisions

- Preserve one shared trailing-normalization structural boundary while keeping separate Layer and
  RMS numerical emitters. This keeps Shape/range/carrier/resource mechanics shared without
  creating a generic run-time normalization interpreter.
- Deduplicate exact repeated logical inputs in first-occurrence order and carry an immutable
  semantic-position map. Generated signatures therefore expose concrete unique carriers while
  preserving semantic operand order and type promotion.
- Use the existing `AGGREGATE_EXACT_STATE` workspace only for Layer's exact-mean pass. Size one
  disjoint slice from the maximum admitted normalized domain for each simultaneous range; RMS
  remains zero-workspace.
- Freeze typed native-order `ValueLayout` constants in generated segment access and omit shared
  invocation-local layout preparation for trailing normalization. This removes preparation work
  from each generated call without changing carrier, alignment, or native-order validation.
- Treat the optimal clean Java implementation as the algorithmic and avoidable-overhead oracle:
  validation, useful arithmetic, pass/dataflow shape, addressing, conversions, and stores match
  the generated cases rather than being padded for the timing gate.

## Known limitations

- Only fully static resolved-layout scalar/parallel-scalar Layer/RMS forward execution is planned.
  Dynamic Shapes, vector/native routes, fusion, partial/combine execution, and batch normalization
  remain unsupported by this task.
- Finite results follow the selected CPU algorithm and tolerance table. Only generated/direct
  identity is raw-bit evidence; no cross-backend bitwise promise is added.
- The performance gate covers only the frozen bounded matrix and environment; it is not a general
  tuning or throughput claim.

## Validation evidence

- Implementation context final CPU validation: `./gradlew :backends:cpu:test --rerun-tasks`
  passed 76 suites/449 tests with zero failures or errors. The focused post-fix validation passed
  10 suites/119 tests with zero failures or errors. This mandatory distinct clean documentation
  context reused those results because it changed comments/Javadocs/documentation only and found
  no stale executable-evidence risk.
- The retained evidence root
  `/private/tmp/synaptik-cpu-0007f-retained-evidence-20260825-final` verifies 241 files through
  `SHA256SUMS`; `SHA256SUMS.sha256` contains digest
  `a7ca999336d73dbf2fee3d2414ff31f0339a5048ec7e2b5ab804b1c5186829c9`. Its summaries record all
  semantic, accuracy, range, alias, canary, and source-identity gates.
- Five isolated forks passed all 85 per-fork and 17 aggregate generated/optimal-clean-Java
  `<= 1.15x` gates. The worst fork ratio was `1.136183168x`; the worst aggregate ratio was
  `1.113266704x`, both BFLOAT16 RMS.
- All 17 retained Java 26 generated classes are final, field-free, constructor-free, and expose
  exactly one typed static `invoke`. Complete `javap -c -p`, `javap -v -p`, member, and opcode
  scans passed; no forbidden generated member or opcode was found. Inspection confirms direct
  three-pass Layer and two-pass RMS bodies, typed constant-layout segment access, and one final
  store per result position.
- Documentation profile: General style plus API/Javadoc, Backend Guide, Planning, and Example
  profiles. This clean documentation context reviewed every changed production Javadoc and test,
  finalized `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan,
  and the roadmap, and made no executable Java change.
- `./gradlew :backends:cpu:javadoc` passed after the final Javadoc edits. The only two warnings are
  the expected Java 26 incubating Vector API module warnings. Rendered pages for the trailing IR,
  lowerer, both emitters, and prepared executable contain the finalized descriptions, parameters,
  results, failures, ownership, range, carrier, and workspace boundaries.
- Final documentation/scope validation passed for Markdown link targets and anchors, balanced
  fences, terminal newlines, whitespace, exact schema/status/history wording, exactly five new
  production types, 35 total changed paths (16 production/Javadoc, 14 tests, and exactly five
  documentation/planning), and empty staging. `git diff --check`, `git diff --cached --check`, and
  `git status --short -uall` supplied the final recorded worktree evidence.
- No-change conclusions: `ARCHITECTURE.md`, focused architecture documentation, ADRs, and
  architecture tests remain correct because ownership, dependencies, and lifecycle boundaries did
  not change. Public Tensor, Compile, and Training APIs and Model/Compiler documentation/contracts
  remain correct because this is CPU-private forward adoption of already-settled first-class
  semantics. Shared Prepare/backend-contract, conformance/integration suites, Gradle/build files,
  and unrelated modules remain unchanged because existing staged resource declaration, cold
  binding, and runtime invocation contracts carry the family without a shared-contract change.

## Implementation notes

- Added exact capability admission for the four first-class forms and a CPU-private
  `CpuTrailingNormalizationIr` plus `CpuTrailingNormalizationLowering` with checked static Shape,
  unique-boundary, access, promotion, epsilon, pass, and Layer-only resource identity.
- Added separate direct `CpuLayerNormEmitter` and `CpuRmsNormEmitter` bodies and the independent
  `CpuTrailingNormalizationReferenceKernel`. Existing generator, route, preparation, finalization,
  executable, and cache seams now carry the new mutually exclusive geometry.
- Layer uses exact represented-value mean, compensated deviation/square sums, and a final
  standardization/optional affine pass. RMS uses scaled sum of squares and a final
  normalization/optional scale pass. Scalar and parallel-scalar ranges own complete leading
  slices and are raw-bit deterministic for identical prepared inputs.
- Carrier binding supports matching typed heap arrays and native-order segments for every unique
  input and output. Static typed layouts replace invocation-local layout construction; cold
  validation still precedes mutation or worker submission. Schema advanced exactly 47 to 48.
- Independent executable review corrected RMS overflow behavior, BFLOAT16 computation boundaries,
  IR invariants, and coverage. Performance review corrected invocation-local native-order layout
  construction and unnecessary shared layout preparation for trailing normalization.
- No package-info file required a change: the existing CPU, IR, lowering, emitter, reference,
  prepare, route, cache, and executable package summaries already describe their stable owners;
  affected type/member Javadocs now carry the family-specific contract.

## Completion summary

- Completed changes: first-class static Layer/RMS CPU admission, lowering, direct generated
  execution, reference verification, schema-48 compatibility, deterministic complete-slice
  orchestration, and finalized documentation/Javadocs.
- Files changed or created: 35 total—16 production/Javadoc, 14 tests, and exactly five
  documentation/planning paths; exactly five new CPU-private production types.
- Tests and validation: reused the final 76-suite/449-test and focused 10-suite/119-test passes;
  retained semantic/Class-File/member/manifest and five-fork performance evidence passes; final
  CPU Javadoc and documentation/scope/whitespace/staging gates pass.
- Documentation-agent review: completed in this mandatory distinct clean documentation-focused
  context; no executable behavior was changed and Java/performance work was not repeated.
- Documentation impact: CPU guide, glossary, task, CPU master plan, and roadmap finalized with
  current/planned boundaries and exact evidence.
- Javadoc review: all 16 affected production paths reviewed; contract-relevant descriptions,
  inputs, results, failures, ownership, layouts, carriers, ranges, and workspace rules finalized.
- Glossary impact: existing Layer/RMS entries now state the stable CPU execution boundary; no new
  glossary heading was needed because normalized Shape, population variance, uncentered mean
  square, root mean square, epsilon, affine transform, and accumulator/computation format already
  have stable definitions.
- Unresolved issues: None for CPU 0007F. CPU 0007A1D remains independently Review needed.
- Follow-up required: None for CPU 0007F. CPU 0007F1 and all later CPU work remain Draft without
  detailed specifications.

Status: Complete
