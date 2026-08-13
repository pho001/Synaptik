# Task 0007: Portable Cumulative Scan Coverage

## Status

Complete

## Goal

Add the first bounded slice of the former broad CPU 0007 frontier: execute exactly one fully
static, resolved-layout current Model `CUM_SUM` or `CUM_PROD` occurrence through the portable
generated-kernel route.

This slice completes the existing `CumulativeScanKind` family across its five numeric data types
and all inclusive/exclusive and forward/reverse modes. Each independent scan slice is traversed
sequentially in logical axis order while distinct slices may execute in parallel. This preserves
one deterministic finite-precision order per output and establishes the range/body seam needed by
later reduction work without mixing aggregate combination, statistics, softmax, or normalization
into the same task.

## Scope

### Exact occurrence and type matrix

- Admit exactly one CPU-owned node whose kind is `CumulativeScanKind.CUM_SUM` or
  `CumulativeScanKind.CUM_PROD`, whose attributes are exact `CumulativeScanAttrs`, and whose
  ordered boundary is one input followed by one output.
- Support FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64. Reject BOOL and every other type.
- Require fully static input/output Shapes, the exact same Shape and data type, resolved layouts
  with non-negative offsets and strides, a valid normalized axis, and an injective output layout.
  The input may use any current supported resolved layout, including offset, positive-strided,
  interleaved, transposed, and zero-strided reads.
- Reject scalar input because no valid scan axis exists. Accept zero extents; no output element is
  written and no worker is submitted when the logical output is empty.
- Keep output and input as distinct complete physical bindings. Reject any complete-span overlap
  before generated execution or worker submission; this task does not introduce in-place scan
  semantics.

### Logical traversal and finite-precision realization

- A scan slice fixes every logical coordinate except the selected axis. Slice ordinals use the
  row-major order of the non-axis coordinates and are independent of physical layout.
- Forward mode visits axis coordinates from zero upward; reverse mode visits them from the last
  coordinate downward without changing output coordinates.
- Inclusive mode updates the accumulator with the current input and then writes it. Exclusive mode
  writes the current accumulator and then updates it. The identities are represented positive zero
  for `CUM_SUM` and represented positive one for `CUM_PROD`.
- INT32 and INT64 use exact-width two's-complement modular addition or multiplication after every
  visited value.
- FLOAT32 and FLOAT64 perform one ordinary same-type IEEE 754 addition or multiplication for each
  visited value and retain that rounded result as the next accumulator. BFLOAT16 widens each exact
  represented input and accumulator to FLOAT32, performs one FLOAT32 operation, and rounds once
  back to BFLOAT16 after every visited value. No reassociation, wider retained accumulator,
  reciprocal rewrite, fused operation, vector prefix algorithm, or cross-slice combination is
  permitted.
- This CPU-private evaluation order must satisfy the Model special-value and identity contract.
  NaN payload choice remains unspecified. Scalar and parallel execution must nevertheless be
  bitwise identical because both execute the same per-slice order.

### Generated execution and resources

- Add focused CPU-private `CpuScanIr`, `CpuScanLowering`, and `CpuScanEmitter` owners in the
  existing IR, lowering, and Class-File emission packages.
- Lower one occurrence to one computation unit and one generated artifact. The boundary list and
  declared buffers are exactly `[input, output]`; there is no virtual result, materialization,
  workspace, partial buffer, carry buffer, or persistent mutable state.
- Generated primitive `start`/`end` bounds denote a contiguous range of independent scan-slice
  ordinals. The generated entry bridges directly to a CPU-owned static scan body, which
  reconstructs the corresponding non-axis coordinates once per slice and walks the selected axis
  without allocating an object per slice or element.
- Scalar execution covers all slice ordinals directly. Parallel-scalar execution partitions only
  the independent slice domain into deterministic disjoint ranges. Never split one scan slice
  across workers. Vector and parallel-vector scan bodies are out of scope.
- When the selected axis is non-empty but the product of the non-axis extents is one, execution is
  scalar. When any non-axis extent or the selected extent is zero, no generated call or worker
  submission occurs.
- Cold lowering computes and validates checked element counts, slice counts, byte spans, axis
  geometry, and address bounds. Cold binding validates carrier/type compatibility, byte sizes,
  alignment, writeability, complete spans, and non-overlap before returning direct typed fields.
- The hot path performs no `Operation`/`CompiledNode` access, graph inspection, layout discovery,
  map lookup, boxing, reflection, string dispatch, synchronization, allocation, division/modulo
  per element, route selection, or cache lookup.

### Capability, compatibility, and evidence

- Capability reporting is true only for the exact occurrence matrix above and remains false for
  all aggregate reductions, arg extrema, masked reductions, statistical reductions, softmax,
  layer/RMS/batch normalization, and unsupported scan rows.
- Advance generated compatibility from schema 19 to schema 20 with no migration reader.
- Structural IR, fingerprint, specialization, and compatibility identity include the scan kind,
  data type, normalized axis role, exclusive/reverse modes, rank, boundary roles/count, access-plan
  structure, finite-precision rule, execution compute mode, carrier pattern, and absence of
  workspace. Concrete extents, offsets, stride magnitudes, slots, carrier objects, addresses,
  worker identity, run identity, and selected range count remain cold when they do not alter
  emitted bytes.
- Extend capability, lowering dispatch, sealed IR permits/encoding, portable route, preparation,
  finalization, generator dispatch/validation, executable binding, scalar reference, schema, and
  focused tests only as required for this family.
- Test generated results against an independent reference for all five types, both kinds, all four
  modes, arbitrary layouts/carriers, empty and singleton axes, modular overflow, signed zero, NaN,
  infinities, and scalar/parallel parity. Reference tests must not invoke production scan helpers.
- After executable Java stabilizes, hand the uncommitted diff and exact CPU-test evidence to a
  separate clean documentation-focused context. That pass finalizes affected Javadocs/package
  summaries, the CPU backend guide, glossary impact, this task evidence, CPU master plan, and
  roadmap.

## Out of scope

- Full, single-axis, multi-axis, masked, target-Shape, arg-extrema, logarithmic, statistical, or
  norm reductions
- softmax, log-softmax, layer normalization, RMS normalization, batch normalization, attention,
  loss, pooling, convolution, or linear algebra
- more than one node, scan fusion, scan/reduction fusion, general partition-DAG decomposition,
  materialized splits, partial scans, cross-worker prefix combination, vector scan, or native routes
- in-place or overlapping input/output execution, negative physical strides, dynamic Shapes or
  layouts, runtime-bound axis geometry, or a shared Runtime scan primitive
- a backend-neutral or cross-backend floating accumulation order, NaN payload promise, bitwise
  result promise, accuracy policy, or public scan configuration
- Model, Compiler, shared Runtime/Prepare, Config, Trace, backend-contract, architecture,
  architecture-test, Gradle/dependency, backend-conformance, integration, Engine, NN, training,
  tuning, benchmark, persistence-policy, or public API changes
- per-element allocation, boxing, reflection, string/map dispatch, synchronization, or a generic
  reduction/scan registry, manager, service, cursor, or interpreter

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0005A partition-kernel reset](0005a-atomic-partition-kernel-architecture-reset.md)
- [CPU 0005C vector and parallel strategies](0005c-vector-and-parallel-portable-strategies.md)
- [CPU 0005D evidence gate](0005d-materialization-specialization-and-persistence-evidence-gate.md)
- [CPU 0005E typed portable family expansion](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
- [CPU 0006 static affine views](0006-portable-static-affine-views-and-boundary-materialization.md)
- [CPU 0006B2 overlap fold](0006b2-portable-overlap-fold.md)
- [CPU 0006C ordering](0006c-portable-stable-ordering-and-selection.md)
- [CPU 0006D explicit-state RNG and dropout](0006d-portable-explicit-state-rng-and-dropout.md)
- [Model 0023E cumulative scan normalization and product](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- Model owns the scan kind, axis, inclusion/direction modes, type domain, exact-width integral
  arithmetic, and abstract special-value meaning. CPU owns only its finite-precision realization,
  private IR, generated route, resources, binding, and execution.
- CPU analysis deterministically lowers one supported occurrence, declares exactly two buffers and
  zero workspace before CPU-blind shared assignment, and finalizes one artifact only after slot
  assignments exist.
- Prepared recipes remain immutable and reusable. Concurrent runs use distinct `RunState` values;
  the scan executable owns no mutable prepared accumulator or shared scratch.
- Runtime receives only a prepared executable with cold-bound direct carrier references and
  primitive geometry. It performs no scan interpretation, graph work, route selection, or
  resource discovery.
- Work remains inside `backends/cpu`, adds no dependency or package, and preserves the current
  generated portable route and optional artifact lifecycle.
- Capability truth may not exceed complete lowering, declaration, assignment, generation,
  binding, overlap validation, execution, and independent reference evidence.
- Stop if exact execution requires a Model/compiler/shared Runtime or Prepare change, if BFLOAT16
  per-step rounding conflicts with the current Model contract, or if a slice cannot be kept within
  one worker without changing the prepared execution boundary.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — sole supported fail-closed capability provider.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — CPU-private structural scan IR.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — one-node validation and cold geometry.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct Class-File scan emission.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`, `.route.portable`, `.cache`,
  `.executable`, `.memory`, and `.reference` — existing staged realization, binding, execution,
  and independent reference boundaries.

Packages added or changed:

- No package is added, removed, moved, or exported. Existing CPU-internal packages gain only the
  focused scan-family types and direct integration described here.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr` — immutable scan semantics,
  boundary roles, numerical order, slice-domain, and compatibility identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLowering` — exact one-node
  revalidation, declarations, access plans, and compact slice/axis geometry.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuScanEmitter` — allocation-free
  generated per-slice traversal, typed accumulation, and output stores.

Tests mirror these production packages. No generic utility package or new public CPU type is
permitted.

## Affected files

Expected production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuScanIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScanLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScanEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

Expected CPU tests:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuScanIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScanLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScanGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferBindingTest.java`

Expected documentation/planning paths during implementation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No Model/compiler/shared Runtime/Prepare/API/architecture/Gradle/conformance/integration path is
expected to change.

## Maximum scope

This task may create or modify at most 42 paths: 24 production/package paths, 13 CPU test paths,
and 5 documentation/planning paths. Stop before a 43rd path, a new package, or any path outside
the listed CPU module and documentation/planning set. A stale existing CPU inventory test may
replace another listed test path but must not increase the ceiling.

If implementation requires a shared contract, public API, another semantic family, another
workspace, partial-scan combination, or more paths, stop and propose a focused follow-up or
architecture decision. Do not hide incomplete scan acceptance behind a follow-up.

## Acceptance criteria

- Capability and lowering admit exactly the one-node static/resolved five-type CUM_SUM/CUM_PROD
  matrix with all four modes and reject every unsupported row truthfully.
- Boundary roles, declarations, assignments, generated parameters, direct bound fields, prepared
  access declarations, and stores are exactly one read input followed by one write output.
- Forward/reverse and inclusive/exclusive traversal, identities, exact-width integral wrap, and
  the specified per-step floating/BFLOAT16 rounding order match this specification.
- Scalar and parallel-scalar execution are bitwise identical for every output because a scan slice
  is never divided between workers.
- Dense, offset, positive-strided, transposed/interleaved, and zero-strided reads; injective
  non-dense writes; heap/native/mixed carriers; empty/singleton axes; modular overflow; finite
  values; signed zeros; NaNs; and infinities agree with an independent reference.
- Complete input/output overlap fails during cold binding before output mutation or worker
  submission, including spans whose physical intersection lies outside a selected range prefix.
- Zero workspace/materialization/scratch, bounded candidates, one realized artifact, immutable
  prepared state, and allocation-free hot loops are enforced by construction and tests.
- Schema 20 rejects schema-19 artifacts. Every semantic/code-shaping scan fact is included in
  deterministic structural/fingerprint/compatibility identity and every listed cold fact remains
  excluded when it does not change emitted bytes.
- Existing pointwise, movement, indexing, scatter, fold, ordering, random, worker, cache, and
  persistence behavior remains unchanged.
- A separate documentation-focused context finalizes affected Javadocs, package summaries, CPU
  guide, glossary impact, planning evidence, and explicit API/architecture/conformance no-change
  conclusions before this task becomes Complete.

## Tests / validation

During implementation, run focused tests for capability, scan IR/lowering/generated execution,
all data types and modes, arbitrary layouts/carriers, preparation/finalization, binding/overlap,
scalar/parallel determinism, schema/cache identity, and independent reference comparison. After
executable Java stabilizes, run one final module suite:

```bash
./gradlew :backends:cpu:test
```

The separate clean documentation pass reuses that evidence unless it changes executable Java and
runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates affected Markdown links and anchors, balanced fences, one terminal newline,
trailing whitespace, generated Javadocs, exact changed-path membership and ceiling, package/type
placement, schema 20, scan matrix and modes, task/master/roadmap status coherence, and absence of
a detailed CPU 0007A task file.

Repository-wide tests, architecture tests, backend conformance, and integration tests are deferred
to CPU 0009 or continuous integration. Run them here only if implementation unexpectedly changes
a repository-wide, dependency, architecture, or reusable cross-backend contract; such a change is
outside scope and normally requires stopping first.

## Dependencies

- CPU 0005A–0006D: Complete, including typed carriers, arbitrary resolved-layout access, generated
  start/end entries, scalar/parallel-scalar workers, cold overlap validation, and schema 19.
- Model 0023E: Complete current `CumulativeScanKind`/`CumulativeScanAttrs` family, five-type domain,
  four modes, identities, integral modular product, and floating special-value semantics.
- Current Compiler capture/inference and shared Prepare/Runtime cold-binding contracts: Complete
  and unchanged; inspected only to confirm the one-input/one-output occurrence and direct prepared
  execution boundary.

## Follow-up tasks

- Draft CPU 0007A owns ordinary aggregate reductions, including full, single-axis, multi-axis,
  and binding-aware SUM forms. It has no detailed specification yet.
- Draft CPU 0007B owns arg-min and arg-max; Draft CPU 0007C owns two-input masked SUM/MEAN; Draft
  CPU 0007D owns advanced logarithmic, statistical, and norm reductions.
- Draft CPU 0007E owns stable SOFTMAX/LOG_SOFTMAX, and Draft CPU 0007F owns layer, RMS, and batch
  normalization after the required aggregate/statistical foundations.
- CPU 0009 retains the portable generated-coverage and conformance checkpoint.

## Architecture impact

Expected impact: None.

This task uses the existing concrete-backend ownership of lowering, route selection, generated
artifacts, storage, and execution. It changes no module responsibility or dependency direction.
If implementation requires architecture, another module, or a shared contract change, stop and
report the exact conflict instead of editing around it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Implement Synaptik CPU task 0007 exactly from its Ready specification. Do not use GSD. Read
AGENTS.md, ARCHITECTURE.md, the current architecture plan, documentation rules and applicable
profiles, planning guide, roadmap, CPU master plan, task 0007, completed CPU tasks
0005A/0005C/0005D/0005E/0006/0006B2/0006C/0006D, Model task 0023E, and every affected or directly
relevant scan/CPU source and test in full before editing.

Deliver one-node static resolved five-type CUM_SUM/CUM_PROD through the generated portable route
with all four modes, exact per-step typed accumulation, BFLOAT16 round-after-each-operation,
logical-slice scalar/parallel determinism, arbitrary supported layouts/carriers, complete
pre-mutation overlap rejection, zero workspace/materialization, independent reference evidence,
truthful capability, and schema 20. Preserve every exclusion and the 42-path ceiling. Stop on any
architecture, shared-contract, Model-semantic, numerical, resource, or scope conflict.

Run focused tests and one final ./gradlew :backends:cpu:test after executable Java stabilizes. Do
not commit or push. Then hand the uncommitted diff and exact Java evidence to a distinct clean
documentation-focused context following docs/developer-guide/documentation-rules.md. That pass
must finalize affected Javadocs/package summaries, CPU guide, glossary impact, task/master/
roadmap, no-change conclusions, Javadoc, Markdown, exact-scope, and whitespace validation without
repeating the successful Java suite unless executable behavior changes. Do not mark Complete until
both passes and every acceptance criterion succeed. Return exact evidence, changed paths,
unresolved issues, context ID if available, and Status: Complete or Status: Incomplete with
required follow-up.
```

## Local decisions

- Select the closed cumulative-scan family first because it is the smallest complete Model family
  in broad CPU 0007, has no dependency on aggregate combination, and can parallelize only across
  independent slices while preserving one exact traversal order.
- Use one typed rounded accumulator per slice. BFLOAT16 rounds after each operation; FLOAT32 and
  FLOAT64 retain their own formats; integral arithmetic wraps at its own width.
- Use slice ordinals for generated and worker ranges. This prevents cross-worker prefix state and
  makes scalar/parallel results identical without workspace.
- Reject input/output overlap rather than introduce an in-place traversal/alias contract that the
  Model and current CPU prepared boundary do not promise.
- Keep vector scan, partial-scan combination, and all aggregate/statistical/normalization families
  outside this first slice.

## Known limitations

- Coverage is one fully static, resolved-layout occurrence with non-negative physical strides.
- Floating bit patterns are stable only for this exact CPU per-step realization and execution
  strategy, not a Model, cross-backend, or future-version promise; NaN payloads are unspecified.
- Parallelism requires at least two independent non-axis slices. One long scan slice remains
  scalar in this task.
- No aggregate reduction, softmax, or normalization capability is advertised by this task.

## Validation evidence

- Planning context `019ffcbc-6b82-7d43-bf99-cb71ef09e77f` produced this bounded task. Initial
  implementation context `019ffcc6-534a-7cb0-a008-3e9dd9f0e5a3` delivered the scan route, and
  first documentation context `019ffcd8-55ac-7482-a08d-4638dae6fa4b` performed the initial
  documentation pass.
- Corrective implementation/audit context `019ffcea-9e5e-7d60-9c45-f5664c8c4d4c` aligned
  capability and lowering on one bounded injectivity decision. Layouts with at most 1,000,000
  elements use exact address enumeration; larger layouts use the bounded monotone-span sufficient
  proof. The focused regression passed 1 suite and 3 tests. It proves that Shape `[3, 2]` with
  strides `[2, 3]` is injective and accepted by capability and lowering, while strides `[1, 1]`
  collide and are rejected.
- After the correction stabilized, the final focused CPU 0007 matrix passed 11 suites and 109
  tests with no skips, failures, or errors. The sole latest authoritative
  `./gradlew :backends:cpu:test` passed 50 suites and 292 tests with one existing opt-in skip and
  no failures or errors. The environment was OpenJDK 26.0.1+8-34 with Gradle wrapper 9.6.1.
- Mandatory clean documentation context `019ffcf5-ac50-7920-b528-1ea57c175e96` independently
  reviewed and finalized the affected contracts without changing executable Java or tests. It
  used the Backend Guide, API/Javadoc, Planning, Guide, Glossary, and Example profiles with the
  general documentation style and reused the stable Java evidence above.
- `./gradlew :backends:cpu:javadoc` passed after final Java/package documentation edits. Javadoc
  reported only the expected two incubating-module warnings for `jdk.incubator.vector`. Rendered
  pages inspected were `CpuScanIr` and `CpuScanIr.Kind`, `CpuScanLowering`,
  `CpuScanLowering.Layout`, `CpuScanLowering.Geometry`, `CpuScanEmitter`,
  `CpuCapabilityProvider`, `CpuPreparedExecutable`, and the CPU root, IR, lowering, emitter,
  cache, portable-route, prepare, executable, and reference package summaries. The rendered
  contracts cover type/mode identity, BFLOAT16 per-step rounding, whole-slice ranges, exact two-
  boundary/no-workspace ownership, complete overlap rejection, and schema 20.
- A targeted deterministic five-file Markdown checker passed local file targets and generated
  heading anchors, balanced backtick/tilde fences, one terminal newline, and trailing whitespace
  for the CPU guide, glossary, task, CPU master plan, and roadmap. Manual review recalculated the
  `[1, 2, 3]` scan example, confirmed terminology against the existing cumulative-scan glossary
  entry, and found no conceptual code presented as runnable API.
- The final CPU-attributable delta is 39 listed paths: 23 production/package paths, 11 CPU test
  paths, and the five permitted documentation/planning paths. The shared glossary and roadmap
  retain concurrent NN hunks, and unrelated NN planning/source/test paths are excluded from the
  CPU count. No CPU 0007A task file exists. The historical schema/status check confirmed CPU 0007 Complete,
  CPU 0007A Draft without a detailed specification, and generated schema 20. `git diff --check`
  passed. No commit or push was performed.

## Implementation notes

- Added focused `CpuScanIr`, `CpuScanLowering`, and `CpuScanEmitter` owners. Lowering revalidates
  exactly one current CUM_SUM/CUM_PROD occurrence, derives checked complete layouts plus an
  independent non-axis slice count, declares `[input, output]`, and retains zero workspace and
  materialization.
- The generated two-boundary entry bridges to a CPU-owned static scalar body. That body
  reconstructs non-axis coordinates once per slice and performs one forward or reverse
  sequential axis traversal. Inclusive mode updates then writes; exclusive
  mode writes then updates. INT32/INT64 wrap naturally, FLOAT32/FLOAT64 retain their own format,
  and BFLOAT16 rounds back after every operation.
- Preparation may choose scalar or parallel-scalar orchestration from the slice count. Each
  generated range owns complete slices only. Empty axes or any empty non-axis domain submit no
  generated call or worker work, and a single slice remains scalar.
- Cold binding validates complete carriers and complete input/output physical spans before any
  generated call or worker submission. The prepared recipe retains direct typed fields and
  immutable cold `CpuScanLowering.Geometry`. For each invocation range, `Geometry.pack` creates a
  fresh mutable `long[]` that owns that range's coordinate state. The CPU-owned scan body inspects
  no Model or graph objects and performs no allocation, route/cache lookup, reflection, boxing,
  synchronization, or per-element division/modulo; generated bytecode is only its direct bridge.
- Added an independent scalar reference implementation and differential evidence across all five
  types, both kinds, and all four modes, plus generated and prepared-execution coverage for
  overflow, signed zero, infinities, NaN, BFLOAT16 per-step rounding, slice parallelism, and
  pre-mutation overlap rejection. Generated compatibility is schema 20.

## Completion summary

- Completed changes: Implemented and documented one static resolved-layout five-type CUM_SUM/
  CUM_PROD occurrence with all four modes, sequential typed accumulation, scalar or whole-slice
  parallel-scalar execution, complete pre-mutation overlap rejection, zero workspace/
  materialization, and schema 20.
- Files changed or created: exactly 23 of the production/package paths listed under Affected
  files (all except unchanged `CpuKernelSpecialization.java`); exactly 11 of the listed CPU tests
  (all except unchanged `CpuClassFileKernelGeneratorTest.java` and `CpuBufferBindingTest.java`);
  and exactly `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task,
  `docs/planning/backends/cpu/master-plan.md`, and CPU-specific roadmap hunks in
  `docs/planning/roadmap.md`. No other path is attributable to CPU 0007.
- Tests and validation: Reused the scope-stable implementation evidence from context
  `019ffcea-9e5e-7d60-9c45-f5664c8c4d4c` (corrective 1 suite/3 tests, focused 11 suites/109 tests,
  and final CPU 50 suites/292 tests, one existing skip, no failures/errors). Final CPU Javadoc,
  rendered-page inspection,
  Markdown/link/anchor/fence/example/terminology/newline/whitespace checks, exact 39-path inventory,
  status/schema/no-0007A-spec checks, and `git diff --check` passed.
- Documentation-agent review: Mandatory separate clean documentation context
  `019ffcf5-ac50-7920-b528-1ea57c175e96` completed the final review. It changed no executable
  behavior or tests.
- Documentation impact: Finalized the CPU guide's ninth-family scan contract, current/planned
  boundaries, and schema-20 summary; synchronized this task, CPU master plan, and CPU roadmap.
- Javadoc review: Finalized the three new scan contracts, affected integration-type summaries,
  constructor/component/failure documentation, and nine materially affected package summaries.
  Existing unaffected member contracts remained accurate and required no mechanical restatement.
- Glossary impact: Updated the existing cumulative-scan entry and implementation-status paragraph
  for the bounded CPU route; no new project term was introduced. Concurrent NN glossary hunks were
  preserved.
- Unresolved issues: None for CPU 0007.
- Follow-up required: None for CPU 0007. CPU 0007A remains Draft without a detailed specification.

Status: Complete
