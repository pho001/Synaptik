# Task 0007A0B: Indexing Generated-Loop Parity

## Status

Complete

## Goal

Replace the bridge-only generated output writers for the completed CPU indexing family with
carrier- and type-specialized loops embedded in each generated class. `GATHER`,
`GATHER_ELEMENTS`, `GATHER_ND`, and `ONE_HOT` must preserve every CPU 0006A2 semantic,
validation, layout, carrier, range, resource, overlap, and lifecycle contract while dense
FLOAT32 `GATHER_ELEMENTS` and BOOL `ONE_HOT` independently reach measurable near-parity with
equivalent direct Java output loops.

This is one cohesive CPU-private correction. All four indexing families share the same
`CpuIndexingIr`, compact geometry, mixed index/value carrier specialization,
`CpuIndexingEmitter`, and generated-artifact compatibility boundary. Splitting them would leave
some current schema-14 indexing identities bridge-only inside the next current schema. The task
changes generated realization only; the completed CPU 0006A2 capability and semantics remain
unchanged.

## Scope

### Reproducible pre-change baseline

- Before any production change, create an isolated `/tmp` probe that measures the current
  generated class against an equivalent direct Java output loop for each required case below.
  Retain the source, exact compile/run/summarize commands, Java/operating-system/architecture
  facts, fixed heap, generated class bytes and disassembly, raw fork samples, summaries, ratios,
  exact verification results, and checksums together.
- Time only the warmed already-validated output-writing pass for both generated and direct forms.
  Exclude lowering, generation, class definition, cold binding, complete index validation,
  allocation, initialization, verification, and output observation from timing. Both timed forms
  must read the same valid index carrier, perform the same coordinate mapping and represented-bit
  writes, and use the same arbitrary-range contract.
- Use these independent dense heap-array cases, each with at least 1,048,576 outputs:
  - FLOAT32 `GATHER_ELEMENTS` on a dense rank-two data/indices/output shape with the selected final
    axis and deterministic valid INT32 indices; and
  - BOOL `ONE_HOT` from dense valid INT32 or INT64 indices with a positive trailing depth chosen
    so the output contains at least 1,048,576 canonical BOOL elements.
- The baseline is defect evidence, not an acceptance result or a universal performance claim. Do
  not choose shapes, direct-loop work, or a threshold after observing results.

### Embedded typed indexing writers

- Change `CpuClassFileKernelGenerator` and `CpuIndexingEmitter` so the generated entry embeds the
  selected indexing family body rather than calling `CpuIndexingEmitter.execute2`,
  `CpuIndexingEmitter.execute3`, or another generic `Object` helper for hot output work.
- Select `GATHER`, `GATHER_ELEMENTS`, `GATHER_ND`, or `ONE_HOT`, data/output type, INT32 versus
  INT64 index load, ordered carrier form, structural access form, and dense/general loop shape at
  generation time from existing typed CPU-private IR and specialization facts. Do not perform a
  per-element `DataType`, family, carrier, array/segment, rank-form, string, or map dispatch.
- Use `CpuCarrierEmitter` or equally focused existing emission owners for direct primitive-array
  and native-order `MemorySegment` instructions. Gather forms copy the exact represented value;
  BFLOAT16 remains opaque 16-bit movement, BOOL gathers preserve the existing canonical-input
  binding rule, and `ONE_HOT` writes exactly byte `0` or byte `1`.
- Preserve the universal primitive `long start, long end` generated entry. Every family must
  honor arbitrary legal half-open output ranges and scalar or parallel-scalar disjoint range
  ownership. A generated class retains no mutable run state, worker, carrier, or geometry.
- Emit no hot-path object allocation, reflection, boxing, generic carrier access, graph or
  Runtime inspection, backend/route/cache selection, or avoidable per-element conversion. Cold
  geometry-array creation and range-start initialization remain outside the generated hot loop.

### Dense and general generated forms

- For cold-proved dense heap-array boundaries, reuse the current
  `DENSE_HEAP_ARRAY_INT` structural proof where sufficient. Narrow universal range bounds, bases,
  and other proved Java-array-index facts once before the loop, then use direct typed integer
  address state without per-element `l2i`, layout-table scans, division, or remainder where the
  family's dense mapping does not require them.
- Dense `GATHER_ELEMENTS` must read the aligned index, replace only the selected data-axis
  coordinate, load one typed data element, and store one typed output element without generic
  helper calls. Dense `ONE_HOT` must compare the trailing class coordinate with the already
  validated row index and store one canonical byte without repeated carrier/type selection.
- Dense `GATHER` and `GATHER_ND` must likewise use typed embedded loops and hoist invariant family,
  rank, axis/tuple, extent, stride, and layout-position facts as far as their structural and cold
  geometry split permits. Do not introduce fixed-shape class identity or an output-sized address
  table to remove legitimate coordinate work.
- Keep a generated typed `GENERAL_LONG` fallback for every supported segment, mixed-carrier,
  arbitrary-layout, non-unit/zero-stride input, offset/strided injective output, large-range, and
  otherwise unproved form. Coordinate division or remainder may occur only during range-start
  initialization or where an arbitrary structural form genuinely requires it; iteration should
  continue with primitive carry/reset state.
- A failed dense proof selects the correct general form and never rejects a case supported by CPU
  0006A2. Preserve empty/scalar domains, scalar indices, batch and tuple-depth forms, deduplicated
  semantic inputs, both index widths, all six gather value types, and heap/segment/mixed carrier
  combinations.

### Validation, resources, and lifecycle

- Preserve `CpuPreparedExecutable.IndexValidation` as the CPU-owned bound execution-time
  validation action. Cold binding resolves its direct INT32/INT64 carrier and compact layout
  geometry; every invocation scans the complete logical index domain in deterministic row-major
  order before any generated call or worker submission.
- Generated output loops consume only already-validated index values. They must not repeat bounds
  checks, normalize negative indices, move validation into Runtime, validate independently per
  worker, or permit any physical output mutation when validation fails.
- Preserve the first-invalid-index exception types, messages, ordinals, selected axes/extents,
  independent empty-index and zero-output domains, including non-empty `GATHER_ND` tuple
  validation with a zero result suffix, and complete no-write behavior on invalid input.
- Preserve exact unique-input declarations, one distinct injective output, zero workspace, zero
  materialization, one unit/artifact, scalar or parallel-scalar output strategy, complete carrier
  accessibility/writability/alignment/size checks, canonical gathered BOOL validation, full-span
  output/input non-overlap, input/input overlap, and run/resource ownership.
- CPU analysis remains deterministic and measurement-free. CPU finalization generates or reuses
  the already selected class only after shared assignment. Runtime continues to invoke the
  immutable prepared executable and performs no indexing interpretation, generation,
  specialization, cache lookup, route selection, tuning, or benchmark-driven mutation.

### Generated compatibility and stable evidence

- Generated bytes change, so advance `CpuGeneratorSchema.CURRENT_VERSION` exactly from `23` to
  `24`. Schema 24 records the embedded typed indexing body and dense/general code-shape boundary
  while retaining existing indexing family, occurrence-map, boundary type/carrier, and structural
  access compatibility facts.
- Treat a schema-23 envelope as an incompatible safe miss. Add no migration reader, converter, or
  compatibility alias. Update all exact current-schema assertions and retain historical
  schema-14 indexing introduction and schema-23 affine/movement history accurately.
- Preserve compatible shape-polymorphic reuse: concrete extents, counts, normalized axis, batch
  count, tuple depth, one-hot depth, layout magnitudes, range/chunk sizes, values, validation
  results, carriers, byte offsets, slots, workers, and run identity remain cold unless emitted
  bytes actually require a new typed structural distinction.
- Add stable class-file/code-shape tests for all four families and representative dense and
  general forms. They must prove embedded typed loads/stores and loops, the intended integer or
  long address form, and absence of `CpuIndexingEmitter.execute2`/`execute3` or any replacement
  hot `Object` bridge.
- Code-shape tests must reject hot allocation, reflection, boxing, string/map dispatch,
  `DataType`/family/carrier switching, graph/Runtime/backend selection, cache lookup, and avoidable
  per-element conversions. Prefer Class-File model assertions over source-text matching, absolute
  byte offsets, or manual `javap` as the lasting gate.
- Retain truthful general-fallback tests. A dense-only code-shape success cannot replace semantic
  evidence for arbitrary layouts, segment/mixed carriers, arbitrary ranges, or parallel output
  ownership.

### Performance acceptance protocol

- Re-run the two isolated cases after production stabilization with at least five fresh JVM
  forks, five warmup batches per fork, and nine randomized measurement rounds per fork. Use
  adaptive batches lasting at least 25 milliseconds, a fixed heap, deterministic inputs, exact
  pre/post output verification, and an observed output sink outside the timed region.
- Report every generated and direct per-fork median plus every ratio. Acceptance is independent
  per case: the median of generated per-fork medians must be no greater than `1.15x` the median of
  the corresponding direct-loop per-fork medians. Do not average the two cases, widen the
  threshold after measurement, or substitute one passing indexing family for the other.
- Keep timing outside Gradle/JUnit. The probe produces observational evidence only and cannot
  select or mutate production strategy, specialization, cache state, Runtime behavior, or a
  later run.
- If either case exceeds `1.15x`, retain `In progress` or `Review needed`, inspect generated and
  just-in-time code shape, and correct the implementation. Environmental invalidity may justify
  rerunning the unchanged protocol; it does not justify completion.

## Out of scope

- Any new Model operation, indexing signature, type, Shape rule, axis/tuple/depth meaning,
  capability row, negative-index policy, default value, sparse one-hot form, or semantics beyond
  completed CPU 0006A2
- Scatter, fold, ordering, random/dropout, pointwise, affine/movement, scan, aggregate, vector
  gather, vectorized one-hot, native-provider, or later semantic-family work
- New public Tensor, Compile, Runtime, Training, Config, Trace, Prepare, or backend API; new route,
  resource, workspace, materialization, in-place behavior, alias policy, or prepared lifecycle
- Fixed-shape class proliferation, unrolling, per-index/per-output tables, algorithm substitution,
  relaxed numerics, public tuning controls, tuning-cache changes, benchmark-selected production
  settings, Runtime measurement, or profiling-driven mutation
- Architecture, module-boundary, dependency, shared-module, Gradle/build, Java-version,
  architecture-test, backend-conformance, integration-test, Engine, other-backend, NN, training,
  or unrelated documentation changes
- Detailed specifications or implementation for CPU 0007A0C–0007A0F or CPU 0007A1; commit, push,
  staging, revert, or modification of unrelated concurrent work

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Runtime/Prepare/Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0006A2 portable Gather and one-hot indexing](0006a2-portable-gather-and-one-hot-indexing.md)
- [CPU 0007A0 generated hot-path parity correction](0007a0-generated-hot-path-parity-correction.md)
- [CPU 0007A0A affine and movement generated-loop parity](0007a0a-affine-and-movement-generated-loop-parity.md)
- [Model 0018O indexing taxonomy](../../../modules/model/tasks/0018o-indexing-taxonomy-and-unstack-normalization.md)
- [Model 0018E Gather-ND semantics](../../../modules/model/tasks/0018e-gather-nd-semantics.md)
- [Model 0019A2 one-hot encoding](../../../modules/model/tasks/0019a2-one-hot-encoding.md)

## Architecture constraints

- Model owns indexing semantics. This task changes only CPU-private generated realization and
  cannot reinterpret or extend any semantic contract.
- Planning selects CPU ownership only. CPU analysis owns deterministic lowering, structural
  specialization, strategy, and exact declarations; CPU finalization owns compatible generation
  or reuse after assignment; Runtime executes the immutable prepared result only.
- Generated hot code receives typed direct carriers and primitive cold geometry. It must not see
  `Operation`, `CompiledNode`, graph objects, backend discovery, route choice, cache policy,
  benchmark evidence, or Runtime semantic state.
- Prepared recipes remain immutable and reusable. Concurrent runs use distinct `RunState` and
  invocation-private validation/geometry state; generated classes retain no mutable run state.
- Complete deterministic pre-write index validation, universal ranges, all current general
  carrier/layout forms, declarations, overlap/resource checks, and lifecycle boundaries remain
  unchanged.
- Performance evidence is observational and cannot alter production selection. Stop if parity
  requires a public/shared contract, architecture, dependency, semantic, route, resource,
  materialization, Runtime-tuning, or broader-module change.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — generated indexing family,
  typed carrier, and dense/general loop emission.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — structural specialization, exact schema,
  artifact identity, and prior-envelope rejection.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — closed indexing family, occurrence map,
  boundary types, and structural access identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — compact static family/layout geometry
  and range-start state.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — deterministic scalar/parallel-scalar
  plan and selected specialization.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold carrier binding, complete
  pre-write validation, overlap/resource checks, geometry packing, and range invocation.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent semantic oracle used by
  differential tests, not by generated production work.

Packages added or changed:

- No package is added, removed, moved, exported, or made supported API. Only existing CPU-internal
  packages may change.

Type placement:

- `CpuIndexingEmitter` continues to own all four generated indexing output mappings and becomes
  the direct typed-body emitter rather than a generic execution helper.
- `CpuClassFileKernelGenerator` passes the existing typed structural indexing facts to that
  emitter; it does not acquire indexing semantics or Runtime state.
- `CpuCarrierEmitter` remains the shared focused owner of primitive-array and native-order segment
  load/store instruction emission.
- `CpuIndexingIr` and `CpuKernelSpecialization` own only code-shaping compatibility facts;
  `CpuIndexingLowering.Geometry` owns concrete cold mapping and range-start facts.
- `CpuPreparedExecutable` continues to own bound validation-before-write ordering and direct
  invocation. Add no dispatcher, registry, manager, facade, utility package, or Runtime helper.

## Affected files

Expected production and package-documentation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuIndexingEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuIndexingIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuIndexingLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

Expected test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedKernelShapeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuIndexingGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuIndexingIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuIndexingLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0007a0b-indexing-generated-loop-parity.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

An expected path may remain unchanged after explicit review. Before substituting any path, record
the replacement and reason in this section, keep it inside the same owning module or listed
documentation area, and remain within the maximum scope.

## Maximum scope

This task may create or modify at most 34 repository paths: 18 production/package paths, 11 test
paths, and 5 documentation/planning paths listed above. Stop and revise this specification or
propose a separately justified follow-up before a 35th path, a new package, an executable path
outside `backends/cpu`, or a documentation path outside the allowlist. Unused path capacity does
not authorize unrelated cleanup.

## Acceptance criteria

- Both required pre-change baselines are recorded before production edits with complete retained
  evidence and no claim that their ratios are universal or already acceptable.
- Generated classes for all four indexing families embed carrier/type/family-specialized output
  loops and contain no `CpuIndexingEmitter.execute2`/`execute3` or equivalent hot `Object` bridge.
- Dense heap-array forms use direct typed integer loop/address state without avoidable
  per-element conversion, dispatch, allocation, or invariant geometry decode; every unsupported
  dense proof retains a correct typed general-long generated fallback.
- GATHER, GATHER_ELEMENTS, GATHER_ND, and ONE_HOT preserve exact CPU 0006A2 outputs across all
  admitted value/index types, scalar and empty domains, batch/tuple/depth variants, general
  layouts, heap/segment/mixed carriers, arbitrary legal ranges, and scalar/parallel-scalar output
  ownership.
- Bound execution still validates the complete logical index domain once in deterministic order
  before any generated call or worker submission. Every invalid case preserves the exact first
  failure and leaves the complete physical output unchanged.
- Stable Class-File tests prove embedded typed dense and general loops and reject helper bridging,
  hot generic dispatch, allocation, reflection, boxing, string/map dispatch, graph/Runtime/backend
  selection, cache lookup, and avoidable per-element conversions without wall-clock assertions.
- Exact declarations, zero workspace/materialization, one unit/artifact, injective distinct
  output, overlap/resource checks, compatible shape-polymorphic reuse, and concurrent-run
  lifecycle remain unchanged.
- Schema advances exactly `23 -> 24`; schema 23 is an incompatible safe miss, no migration reader
  exists, and all exact schema/key/class-byte assertions pass.
- Dense FLOAT32 GATHER_ELEMENTS and BOOL ONE_HOT each pass the unchanged independent `<= 1.15x`
  median-of-fork-medians gate under the complete five-fork protocol.
- No public API, capability, semantic, route, resource, materialization, architecture, dependency,
  shared-module, Runtime-tuning, build, conformance, integration, later-task, or unrelated change
  is present.
- A distinct clean documentation-focused context independently reviews the final implementation
  diff and evidence, finalizes affected Javadocs/package summaries, CPU guide, glossary impact,
  this task, CPU master plan, and roadmap, and records explicit no-change conclusions where an
  expected path remains unchanged before the task becomes `Complete`.

## Tests / validation

Run focused tests for every changed test owner during implementation. The minimum focused matrix
must include indexing emission/semantics, bound validation-before-write, lowering/IR identity,
preparation/finalization, reference differential behavior, generated class shape, specialization,
and schema/persistence compatibility. After executable Java stabilizes, run once:

```bash
./gradlew :backends:cpu:test
```

Run the isolated generated/direct probe separately from Gradle/JUnit with the exact retained
commands. Both cases must have at least 1,048,576 outputs and use five fresh JVM forks, five
warmup batches, nine randomized rounds, adaptive batches of at least 25 milliseconds, a fixed
heap, deterministic inputs, exact pre/post verification, and a sink outside timing. Assert each
`<= 1.15x` median-of-fork-medians result in the evidence report, not in an ordinary unit test.

The distinct clean documentation pass reuses successful Java and probe evidence unless it
changes executable behavior or the probe, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates rendered affected Javadocs, local Markdown links and anchors, balanced fences,
terminal newlines, trailing whitespace, exact changed-path allowlist/count, package/type
placement, exact schema-24 history and prior-envelope miss, stable bytecode claims, retained probe
protocol/ratios/checksums, task/master/roadmap synchronization, and exact
0007A0B–0007A0F/0007A1 ordering.

Repository-wide, architecture, backend-conformance, and integration validation remains deferred
to CPU 0009 or continuous integration because this task changes one concrete backend and no
shared or architecture contract. Stop and replan if implementation makes that conclusion false.

## Dependencies

- [CPU 0006A2](0006a2-portable-gather-and-one-hot-indexing.md) is `Complete` and owns the exact
  four-family semantics, compact geometry, complete pre-write validation, general
  layout/carrier/range behavior, resource declarations, reference oracle, and schema-14 history
  preserved by this correction.
- [CPU 0007A0](0007a0-generated-hot-path-parity-correction.md) is `Complete` and supplies the
  shared typed carrier emission, dense integer loop/address proof, stable code-shape testing, and
  five-fork performance protocol.
- [CPU 0007A0A](0007a0a-affine-and-movement-generated-loop-parity.md) is `Complete` and establishes
  current schema 23 plus the immediately preceding dense/general generated-loop correction.
- Current Java 26 Class-File API, shared Prepare/Runtime, CPU worker, direct carrier binding, and
  generated-artifact cache contracts are complete and unchanged.

## Follow-up tasks

- CPU 0007A0C remains the next `Draft` row and owns scatter generated-loop parity only after this
  task completes; no detailed specification is created by this task.
- CPU 0007A0D–0007A0F remain ordered `Draft` corrections for fold, ordering, and random/dropout.
- CPU 0007A1 remains `Draft` and explicitly depends on CPU 0007A0F before semantic-family work
  resumes.
- CPU 0009 retains the generated-coverage and conformance checkpoint.

## Architecture impact

Expected impact: None.

The task changes only CPU-owned generated indexing code shape and current-only artifact
compatibility under the existing analysis/finalization/binding lifecycle. If implementation
requires an architecture, public/shared contract, module, dependency, semantic, route, resource,
materialization, or Runtime-policy change, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Implement Synaptik CPU task 0007A0B exactly from its Ready specification. Use a separate clean
implementation context. Do not use GSD. Do not commit, push, stage, revert, or modify unrelated
work.

Read AGENTS.md, ARCHITECTURE.md, the architecture/performance/runtime-boundary documents,
documentation rules and applicable profiles, planning guide, roadmap, CPU master plan, task
0007A0B, completed tasks 0006A2/0007A0/0007A0A, and every affected or directly relevant current
CPU source/test file before editing. Record both required baselines before production changes,
then implement the embedded typed indexing loops, exact schema 24 compatibility, stable code-shape
and semantic tests, one final CPU suite, and the unchanged isolated five-fork per-case parity
gates. Preserve complete bound validation before every write. Stop on any architecture, scope,
semantic, shared-contract, route, resource, lifecycle, or performance conflict.

After executable Java and probe evidence stabilize, hand the uncommitted diff and exact evidence
to a distinct clean documentation-focused context following the documentation rules. That context
must independently inspect the code/tests and finalize affected Javadocs/package summaries, CPU
guide, glossary impact, task/master/roadmap, rendered pages, Markdown, exact scope, schema,
status/order, performance evidence, and whitespace without repeating successful Java or timing
runs unless executable/probe behavior changes or a concrete stale-evidence risk is recorded. Do
not mark Complete until both contexts and every acceptance criterion succeed.

Return exact changed paths, commands/results, environment and per-fork/per-case ratios,
documentation/Javadoc/glossary impact and no-change conclusions, context IDs, unresolved issues,
and one of these exact completion markers:

Status: Complete

or

Status: Incomplete
Follow-up required: <specific follow-up>
```

## Local decisions

- Keep all four indexing families together because they share one bridge-only emitter, typed
  mixed-boundary IR, compact geometry, validation/write split, and schema boundary. Scatter stays
  separate because it owns reductions, duplicates, output-domain scans, and optional scratch.
- Measure the already-validated writer rather than the full bound invocation. This isolates the
  bridge defect while preserving production's mandatory validation pass and comparing identical
  valid-index mapping work in generated and direct loops.
- Require separate FLOAT32 GATHER_ELEMENTS and BOOL ONE_HOT gates because aligned gather copying
  and indicator expansion have different control flow; neither passing case proves the other.
- Advance directly to schema 24 because generated indexing bytes necessarily change. Preserve
  schema-14 introduction and schema-23 history, reject schema 23 without migration, and keep
  concrete geometry cold for shape-polymorphic reuse.
- Keep the dense/general split. Dense heap arrays use proved integer addressing; segment, mixed,
  arbitrary-layout, large-range, and otherwise unproved cases retain typed long state.

## Known limitations

- Performance gates cover two representative large dense heap-array cases on the recorded local
  environment. They do not claim parity for every indexing family, shape, carrier, layout, range,
  index width, CPU, or Java virtual machine.
- General segment/mixed/arbitrary-layout forms receive semantic and stable code-shape validation
  but no timing threshold in this task.
- Complete index validation intentionally remains a separate scalar execution-time pass and the
  generated writer reads valid indices again. This accepted CPU 0006A2 boundary is not redesigned
  here.
- Only the existing one-node fully static resolved-layout CPU indexing capability is corrected;
  multi-node fusion, dynamic layouts, vector indexing, native routes, and new semantics remain
  outside scope.

## Validation evidence

- Clean planning context `01a0002e-0a9f-7133-ba6f-80d2a1515d78` read the governing repository,
  architecture, performance, Runtime/Prepare/backend, planning, and documentation instructions;
  the CPU master plan and roadmap; completed CPU 0006A2/0007A0/0007A0A tasks; directly relevant
  indexing lowering/IR/emitter/generator/specialization/cache/prepare/executable/reference source
  and tests; carrier emission; package summaries; CPU guide; and glossary. The worktree was clean
  before planning edits and contained no narrower `AGENTS.md`.
- Source inspection confirmed `CpuClassFileKernelGenerator` routes indexing to
  `CpuIndexingEmitter.emit`, which currently emits only `execute2(Object,...)` or
  `execute3(Object,...)`. The helper then selects family, `DataType`, carrier, layout positions,
  and array versus segment access around its output loop.
- Source inspection confirmed `CpuPreparedExecutable` creates a bound `IndexValidation`, and
  `Invocation.executeBound()` calls its complete deterministic `validate()` before the first
  generated call or worker submission. Indexing plans retain zero workspace/materialization and
  pack one invocation-private geometry array per output range.
- A read-only temporary Java 26 probe generated current representative GATHER_ELEMENTS and ONE_HOT
  classes. `javap -c -p` showed each entry contains only typed argument loads followed by one
  `invokestatic` to `CpuIndexingEmitter.execute3(Object,...)` or
  `CpuIndexingEmitter.execute2(Object,...)`. The classes were 518 and 465 bytes with SHA-256
  `e5b7faf56f3577a2dcf9a93b70c62478f51ec9a7092328a1ac6f8ddde16fda81` and
  `568fc784794559e3b912b559e07d8d9f9fe6d29d6ac3b2ac6fcfb134dd72586c`, respectively.
- Planning-only validation passed after task/master/roadmap synchronization: the canonical 20
  task headings and single `Ready` status; 450 local Markdown targets across the exact three
  changed Markdown files; balanced fences; one terminal newline; no trailing whitespace; exact
  ordered Complete/Ready/Draft dependencies for 0007A0A–0007A0F and 0007A1; absence of detailed
  0007A0C–0007A0F and 0007A1 specifications; exact changed-path restriction to this task, the CPU
  master plan, and the roadmap; tracked and new-file `git diff --check` equivalents. No Gradle,
  Javadoc, performance, or repository test command was required for this planning-only change.
- Implementation context `01a00039-8814-7e62-a0cd-45df202c6a13` recorded the required pre-edit
  baseline under `/tmp/synaptik-cpu-0007a0b-baseline` before changing production Java. The
  retained protocol used OpenJDK 26.0.1+8-34 on macOS 26.5.2 arm64, fixed `-Xms1g -Xmx1g`,
  1,048,576 outputs per case, deterministic valid INT32 indices, five fresh forks, five randomized
  warmup batches, nine randomized measurement rounds, adaptive batches of at least 25 ms, exact
  pre/post verification, and an observed sink outside timing. The directory retains source,
  commands, environment, raw samples, summaries, class bytes, `javap`, and a verified checksum
  manifest.
- Baseline per-fork generated/direct ratios were GATHER_ELEMENTS `24.087991`, `24.629469`,
  `24.960989`, `24.360516`, `23.963102` and ONE_HOT `16.899260`, `16.983090`, `17.486402`,
  `16.849451`, `17.060267`. Median-of-fork-medians ratios were `24.629260x` and `17.010109x`.
  Baseline generated class SHA-256 values were
  `b808b7e7559451ae96113aaec0921b127e0b5335ddd1a814f35fbe132e898e0e` for GATHER_ELEMENTS
  and `ce33604d3d3b99d28c2d698ad39cf6905c4257748e76338688a59603436016b0` for ONE_HOT;
  disassembly showed only the expected `execute3(Object,...)` and `execute2(Object,...)` bridges.
- An initial post-edit probe under `/tmp/synaptik-cpu-0007a0b-final` found GATHER_ELEMENTS at
  `1.224155x` and therefore did not qualify as final acceptance evidence. The implementation
  replaced its per-element row-coordinate carry with row-fragment inner loops, then ran the
  unchanged protocol from scratch under `/tmp/synaptik-cpu-0007a0b-final-pass`.
- Passing final per-fork generated/direct ratios were GATHER_ELEMENTS `1.082191`, `1.084234`,
  `1.074675`, `1.080400`, `1.101749` and ONE_HOT `0.893464`, `0.895060`, `0.892097`,
  `0.891246`, `1.033875`. Median-of-fork-medians ratios were independently `1.084234x` and
  `0.895060x`, both within `<= 1.15x`. Final class SHA-256 values were
  `a410577acfd772209674b7ee35a3ff3fd0268c52e4cda0f051c9bbdba7b6f081` and
  `f1cf06521ee63b5ceb86d1ed21e6ece14343f0d6260ed695edb7f83dc5648e33` respectively. Final
  `javap` inspection found typed entry signatures and no invocation, allocation, cast, division,
  or remainder instruction in either measured class; the final checksum manifest verified.
- Iteration used focused indexing generation, lowering/IR, executable validation, reference,
  preparation/finalization, specialization, and persistence tests. The final combined focused
  command over the exact 11 listed test owners passed 11 suites/100 tests with zero failures,
  errors, or skips. After executable Java stabilized, the single authoritative
  `./gradlew :backends:cpu:test` passed 53 suites/310 tests with zero failures, zero errors, and
  one existing opt-in persistence skip. Executable Java did not change afterward.
- Stable Class-File assertions now cover every indexing family, dense primitive-array and general
  segment code, direct typed loads/stores, loops, long general arithmetic, absence of bridge or
  `Object` descriptors, restricted segment-call owners, and absence of object allocation.
  Schema assertions require exactly 24 and the persistence corruption matrix uses schema 23 as
  the incompatible previous envelope. `git diff --check` passed after the implementation record.

## Implementation notes

- Implementation context `01a00039-8814-7e62-a0cd-45df202c6a13` replaced the two generic
  `Object` bridge entries with generated typed loops for all four indexing families. Dense
  heap-array ONE_HOT and final-axis GATHER_ELEMENTS use dedicated integer-address loops; other
  dense forms use integer mapped loops and general segment/mixed/layout forms retain typed long
  mapping and carry/reset state.
- The final-axis dense GATHER_ELEMENTS writer groups each arbitrary range into row fragments so
  the inner loop performs one direct INT32 index load, one FLOAT32 data load, and one FLOAT32
  store without a per-element coordinate-wrap branch. This was required to pass the unchanged
  independent performance gate while keeping concrete extents and normalized axis cold.
- `CpuPreparedExecutable.IndexValidation` and its placement before generated calls or worker
  submission were not changed. Existing executable tests continue to cover deterministic
  first-invalid diagnostics, independent empty-index/zero-output domains, and no physical output
  mutation on validation failure.
- Generated compatibility is current-only schema 24. The persistence regression now writes a
  schema-23 envelope and verifies an incompatible safe miss; no migration reader was added.
- Documentation context `01a00074-37e3-7c33-af1c-18233eae156e` independently inspected the final
  source/test diff, bound validation ordering, schema assertions, retained baseline and accepted
  final-pass artifacts, generated class disassembly, package contracts, CPU guide, glossary, and
  planning state. It finalized the three directly affected type Javadocs, cache and emitter
  package summaries, guide/glossary schema-24 explanations, and synchronized task/master/roadmap
  records without changing executable behavior.
- No expected path was substituted. The final 15-path task scope consists of five production or
  package-documentation paths, five test paths, and the five allowed documentation/planning paths.
  No concurrent NN path is part of CPU 0007A0B or was edited by the documentation pass.

## Completion summary

- Completed changes: embedded direct carrier/type/family-specialized generated indexing loops,
  preserved bound validation-before-write, advanced schema 23 to 24, added stable Class-File
  shape coverage, and passed both independent performance gates.
- Files changed or created by the complete task: `CpuGeneratorSchema.java`, cache
  `package-info.java`, `CpuClassFileKernelGenerator.java`, `CpuIndexingEmitter.java`, emitter
  `package-info.java`, `CpuGeneratedKernelArtifactStoreTest.java`,
  `CpuKernelSpecializationTest.java`, `CpuIndexingGeneratedKernelTest.java`,
  `CpuPartitionFinalizerTest.java`, `CpuPartitionPreparerTest.java`, `cpu-backend.md`,
  `glossary.md`, this task record, the CPU master plan, and the roadmap.
- Files changed by documentation context `01a00074-37e3-7c33-af1c-18233eae156e`:
  `CpuGeneratorSchema.java`, cache `package-info.java`, `CpuClassFileKernelGenerator.java`,
  `CpuIndexingEmitter.java`, emitter `package-info.java`, `cpu-backend.md`, `glossary.md`, this
  task record, the CPU master plan, and the roadmap. Java edits in this pass were Javadoc-only.
- Tests and validation: the 11-owner focused matrix passed 11 suites/100 tests with zero
  failures, errors, or skips. The one authoritative `./gradlew :backends:cpu:test` passed 53
  suites/310 tests with zero failures, zero errors, and one existing opt-in persistence skip.
  Both final ratios passed. Independent checksum verification passed for every baseline and
  accepted final-pass manifest entry; the earlier `/tmp/synaptik-cpu-0007a0b-final` investigation
  remains explicitly non-acceptance evidence. The accepted class disassembly contains typed entry
  signatures and no invocation, allocation, cast, division, or remainder instruction.
- Documentation validation: `./gradlew :backends:cpu:javadoc` passed with two incubating-module
  warnings and no Javadoc error. Rendered pages for `CpuGeneratorSchema`,
  `CpuClassFileKernelGenerator`, `CpuIndexingEmitter`, and both changed package summaries were
  inspected. Focused checks passed for local Markdown targets and anchors, balanced fences,
  terminal newlines, trailing whitespace, schema/status/dependency ordering, absence of detailed
  0007A0C–0007A0F specs, exact 15-path allowlist/count, and `git diff --check`. Successful Java
  tests were not rerun because this pass changed no executable behavior.
- No-change conclusions: `ARCHITECTURE.md`, architecture tests, backend conformance, integration,
  build configuration, shared modules, public capability/API, and Runtime/Prepare ownership need
  no change because this task changes only CPU-private generated code shape and current-only
  artifact compatibility. The CPU root/internal, executable, IR, lowering, prepare, and reference
  package summaries remain accurate; their capability, ownership, geometry, validation, and oracle
  contracts did not change. `CpuPreparedExecutable`, indexing IR/lowering/reference,
  `CpuCarrierEmitter`, preparation code, and their Javadocs remain accurate without edits.
- Unresolved issues: none. CPU 0007A0C remains the single next `Draft` frontier, with 0007A0D–
  0007A0F ordered `Draft` and CPU 0007A1 held behind 0007A0F.

Status: Complete
